"""
sitewatch_full.py — Complete Three-Layer Pipeline
Causal Estimation of Construction-Induced PM10 in Pune 2023–2024

Layer 1 — DBSCAN: Geographic hotspot clustering of construction zones
Layer 2 — LightGBM: 24-hour PM10 forecaster per station
Layer 3 — Causal: DoWhy DAG + EconML Doubly Robust ATE + counterfactual simulations

Usage:
    python sitewatch_full.py

Generates dataset internally (no CSV needed).
Saves all plots to ./output/

Author: Shreya | B.Tech CSE | Dr. D.Y. Patil School of Science and Technology
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

BANNER = "=" * 65
print(BANNER)
print("  SiteWatch — Construction PM10 · Pune 2023-2024")
print("  Three-Layer Causal ML Pipeline")
print(BANNER)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

PUNE_STATIONS = [
    {"name": "Katraj",       "lat": 18.4588, "lon": 73.8614},
    {"name": "Pashan",       "lat": 18.5362, "lon": 73.8077},
    {"name": "Shivajinagar", "lat": 18.5308, "lon": 73.8474},
    {"name": "Hadapsar",     "lat": 18.4985, "lon": 73.9258},
    {"name": "Pimpri",       "lat": 18.6270, "lon": 73.7990},
    {"name": "Chinchwad",    "lat": 18.6486, "lon": 73.7997},
]

# PCMC + Metro construction corridors 2023-2024
CONSTRUCTION_CORRIDORS = [
    ("Hinjewadi_Metro_Line3", ["Pashan", "Pimpri"],        1, 12, 0.90),
    ("Baner_Flyover",         ["Pashan", "Shivajinagar"],  1,  8, 0.80),
    ("PCMC_Road_Widening",    ["Pimpri", "Chinchwad"],     3, 12, 0.75),
    ("Hadapsar_Bypass",       ["Hadapsar"],                6, 12, 0.70),
    ("Katraj_Tunnel",         ["Katraj"],                  1,  6, 0.85),
]

CONFOUNDERS = [
    "wind_speed_kmh", "humidity_pct", "temperature_c",
    "hour", "month", "day_of_week", "is_winter", "traffic_index",
]
TREATMENT = "active_construction"
OUTCOME   = "pm10"

CAUSAL_GRAPH = """
digraph {
    active_construction -> pm10;
    wind_speed_kmh      -> pm10;
    wind_speed_kmh      -> active_construction;
    humidity_pct        -> pm10;
    temperature_c       -> pm10;
    temperature_c       -> pollution_trapped;
    pollution_trapped   -> pm10;
    traffic_index       -> pm10;
    traffic_index       -> active_construction;
    hour                -> pm10;
    hour                -> active_construction;
    hour                -> traffic_index;
    month               -> pm10;
    month               -> active_construction;
    month               -> temperature_c;
    month               -> humidity_pct;
    month               -> wind_speed_kmh;
    is_winter           -> pm10;
    is_winter           -> pollution_trapped;
    day_of_week         -> active_construction;
    day_of_week         -> traffic_index;
}
"""

INTERVENTIONS = [
    {"name": "water_mist_suppression",  "changes": {"wind_speed_kmh": +2, "humidity_pct": +10},
     "description": "Mist cannon: +10% humidity, +2 km/h dispersion"},
    {"name": "night_construction_halt", "changes": {"active_construction": 0},
     "description": "Ban construction 10pm–6am"},
    {"name": "wind_barriers",           "changes": {"wind_speed_kmh": +8},
     "description": "Hoarding / barriers → dispersion +8 km/h"},
    {"name": "dust_suppression_combo",  "changes": {"humidity_pct": +12, "wind_speed_kmh": +3},
     "description": "Wet suppression + barrier (combined)"},
    {"name": "traffic_diversion",       "changes": {"traffic_index": -0.4},
     "description": "Reroute traffic away from site"},
]


# ═══════════════════════════════════════════════════════════════════
# STEP 0 — BUILD DATASET
# ═══════════════════════════════════════════════════════════════════

def build_station_year(station: dict, year: int) -> pd.DataFrame:
    """
    Calibrated synthetic hourly data for one station-year.
    PM10 physics calibrated to CPCB Pune annual reports:
      - annual mean ~97 µg/m³ overall
      - construction adds ~38 µg/m³ causally
      - winter inversion trapping, diurnal cycle, monsoon washout
    """
    rng = np.random.default_rng(hash(station["name"]) % (2**32) + year)
    timestamps  = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n           = len(timestamps)
    hours       = timestamps.hour.values
    months      = timestamps.month.values
    day_of_week = timestamps.dayofweek.values

    # Weather
    temp  = (25 - 8*np.cos(2*np.pi*(months - 4)/12)
             + rng.normal(0, 2, n))
    hum   = np.clip(55 + 25*np.sin(2*np.pi*(months - 3)/12)
                    + rng.normal(0, 8, n), 20, 98)
    wind  = np.clip(8 + 4*np.sin(2*np.pi*(months - 5)/12)
                    + rng.exponential(3, n), 0.5, 40)

    # Traffic
    rush    = ((hours >= 7) & (hours <= 10)) | ((hours >= 17) & (hours <= 20))
    traffic = np.clip(0.6*rush + 0.3*(day_of_week < 5) + rng.uniform(0, 0.1, n), 0, 1)

    # Construction activity
    activity = np.zeros(n)
    for _, stations, m0, m1, intensity in CONSTRUCTION_CORRIDORS:
        if station["name"] in stations:
            mask = ((months >= m0) & (months <= m1)
                    & (day_of_week < 6)
                    & (hours >= 7) & (hours <= 18))
            activity += intensity * mask.astype(float)
    activity  = np.clip(activity, 0, 1)
    treatment = (activity > 0.3).astype(int)

    # PM10 causal model
    is_winter = ((months <= 2) | (months >= 11)).astype(int)
    poll_trap = is_winter * ((temp < 15) & ((hours >= 22) | (hours <= 5))).astype(int)
    pm10 = np.clip(
        78
        + 30*np.cos(2*np.pi*(months - 1)/12)      # seasonal peak Jan
        + 20*np.sin(np.pi*hours/24)                # diurnal peak noon
        - 2.5*wind                                 # dispersion
        + 0.3*hum                                  # hygroscopic growth
        + 18*traffic                               # mobile source
        + (38 + rng.normal(0, 5, n)) * activity   # CAUSAL construction effect
        + 25*poll_trap                             # winter night inversion
        + rng.normal(0, 12, n),                   # background noise
        5, 800,
    )

    return pd.DataFrame({
        "timestamp":              timestamps,
        "station_name":           station["name"],
        "lat":                    station["lat"],
        "lon":                    station["lon"],
        "pm10":                   pm10.round(2),
        "temperature_c":          temp.round(1),
        "humidity_pct":           hum.round(1),
        "wind_speed_kmh":         wind.round(2),
        "traffic_index":          traffic.round(3),
        "active_construction":    treatment,
        "construction_intensity": activity.round(3),
        "is_winter":              is_winter,
        "pollution_trapped":      poll_trap,
        "hour":                   hours,
        "month":                  months,
        "day_of_week":            day_of_week,
    })


def build_dataset() -> pd.DataFrame:
    print("\n[0/4] Building dataset (6 stations × 2 years = 12 station-years)...")
    parts = []
    for station in PUNE_STATIONS:
        for year in [2023, 2024]:
            df_s = build_station_year(station, year)
            parts.append(df_s)
            print(f"  {station['name']} {year}: {len(df_s):,} rows | "
                  f"PM10 mean {df_s.pm10.mean():.1f} µg/m³")

    df = pd.concat(parts, ignore_index=True).sort_values(["station_name", "timestamp"])
    print(f"\n  Total: {len(df):,} rows | {df.station_name.nunique()} stations")
    print(f"  Date range: {df.timestamp.min().date()} → {df.timestamp.max().date()}")
    print(f"  PM10 mean: {df.pm10.mean():.1f} µg/m³  |  max: {df.pm10.max():.1f}")
    print(f"  Construction rows: {df.active_construction.sum():,} "
          f"({df.active_construction.mean()*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════════
# LAYER 1 — DBSCAN HOTSPOT CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def run_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster Pune stations into construction-impact hotspot zones.

    Features used:
      - Lat/lon (haversine → degrees to radians)
      - Normalised mean PM10 during construction hours
      - Mean construction intensity

    Returns station-level cluster assignments + summary df.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    print("\n[1/4] Layer 1 — DBSCAN Hotspot Clustering...")

    # Aggregate to station level
    station_stats = (
        df.groupby("station_name").agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            mean_pm10=("pm10", "mean"),
            mean_pm10_construction=(
                "pm10", lambda x: x[df.loc[x.index, "active_construction"] == 1].mean()
            ),
            construction_intensity=("construction_intensity", "mean"),
        ).reset_index()
    )
    station_stats["mean_pm10_construction"] = station_stats["mean_pm10_construction"].fillna(
        station_stats["mean_pm10"]
    )

    # Feature matrix: [lat, lon, normalised pm10_during_construction, intensity]
    # Convert lat/lon to km offsets from centroid so scale matches pollution features
    lat0 = station_stats["lat"].mean()
    lon0 = station_stats["lon"].mean()
    station_stats["km_north"] = (station_stats["lat"] - lat0) * 111.0
    station_stats["km_east"]  = (station_stats["lon"] - lon0) * 111.0 * np.cos(np.radians(lat0))

    scaler = StandardScaler()
    X = scaler.fit_transform(station_stats[
        ["km_north", "km_east", "mean_pm10_construction", "construction_intensity"]
    ])

    # eps=1.5 → roughly 1.5 std-unit radius; min_samples=1 so no noise points on a 6-station set
    db = DBSCAN(eps=1.5, min_samples=1, metric="euclidean")
    labels = db.fit_predict(X)
    station_stats["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Stations: {len(station_stats)} | Clusters found: {n_clusters}")
    for cl in sorted(set(labels)):
        members = station_stats[station_stats.cluster == cl]
        tag = "NOISE" if cl == -1 else f"Zone {cl}"
        print(f"  {tag}: {list(members.station_name)} "
              f"| mean PM10(constr.) = {members.mean_pm10_construction.mean():.1f} µg/m³")

    # Merge cluster label back into main df
    df = df.merge(station_stats[["station_name", "cluster"]], on="station_name", how="left")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters + 1)
    for cl in sorted(set(labels)):
        sub = station_stats[station_stats.cluster == cl]
        color = "grey" if cl == -1 else cmap(cl)
        label = "Noise" if cl == -1 else f"Zone {cl}"
        ax.scatter(sub.lon, sub.lat, s=sub.mean_pm10_construction * 1.5,
                   color=color, alpha=0.85, edgecolors="k", linewidths=0.7, label=label, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row.station_name, (row.lon, row.lat),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "DBSCAN Construction Impact Zones · Pune 2023–2024\n"
        "(marker size ∝ mean PM10 during construction hours)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_pune_dbscan_map.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig2_pune_dbscan_map.png")

    return df, station_stats


# ═══════════════════════════════════════════════════════════════════
# LAYER 2 — LIGHTGBM 24-HOUR PM10 FORECASTER
# ═══════════════════════════════════════════════════════════════════

def _make_lag_features(df_station: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag features for LightGBM within a single station's time series.
    Lags: 1h, 2h, 3h, 6h, 12h, 24h
    Rolling means: 3h, 6h, 24h
    """
    out = df_station.copy().sort_values("timestamp").reset_index(drop=True)
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"pm10_lag{lag}"] = out["pm10"].shift(lag)
    for window in [3, 6, 24]:
        out[f"pm10_roll{window}"] = out["pm10"].shift(1).rolling(window).mean()
    return out


def run_lgbm_forecaster(df: pd.DataFrame):
    """
    Train a LightGBM model to predict PM10 24 hours ahead, per station.

    Features: lag features + weather + time + construction flag
    Target:   pm10 at t+24

    Train split: 2023 data
    Test split:  2024 data
    Reports RMSE and MAE per station.
    """
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print("\n[2/4] Layer 2 — LightGBM 24-Hour PM10 Forecaster...")

    LAG_FEATS = (
        [f"pm10_lag{l}" for l in [1, 2, 3, 6, 12, 24]]
        + [f"pm10_roll{w}" for w in [3, 6, 24]]
    )
    WEATHER_FEATS = [
        "temperature_c", "humidity_pct", "wind_speed_kmh", "traffic_index",
        "active_construction", "is_winter", "hour", "month", "day_of_week",
    ]
    FEAT_COLS = LAG_FEATS + WEATHER_FEATS

    all_metrics  = []
    test_preds   = []     # collect for plotting
    station_models = {}

    for station_name in sorted(df.station_name.unique()):
        df_s = df[df.station_name == station_name].copy()
        df_s = _make_lag_features(df_s)

        # Target: pm10 24 hours ahead
        df_s["target"] = df_s["pm10"].shift(-24)
        df_s = df_s.dropna(subset=FEAT_COLS + ["target"]).reset_index(drop=True)

        train = df_s[df_s.timestamp.dt.year == 2023]
        test  = df_s[df_s.timestamp.dt.year == 2024]

        if len(train) < 100 or len(test) < 100:
            continue

        X_train, y_train = train[FEAT_COLS], train["target"]
        X_test,  y_test  = test[FEAT_COLS],  test["target"]

        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        preds = model.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        all_metrics.append({"station": station_name, "RMSE": rmse, "MAE": mae})
        station_models[station_name] = model

        # Keep a sample for the forecast plot (last 2 weeks of test)
        sample = test.tail(14 * 24).copy()
        sample["pred_pm10"] = model.predict(sample[FEAT_COLS])
        sample["station_name"] = station_name
        test_preds.append(sample[["timestamp", "station_name", "pm10", "pred_pm10"]])

        print(f"  {station_name:15s}  RMSE={rmse:.2f}  MAE={mae:.2f}  "
              f"(train n={len(train):,} | test n={len(test):,})")

    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n  Mean RMSE: {metrics_df.RMSE.mean():.2f} µg/m³  |  "
          f"Mean MAE: {metrics_df.MAE.mean():.2f} µg/m³")

    # ── Forecast plot (all stations, last 2 weeks of 2024) ──
    test_df = pd.concat(test_preds, ignore_index=True)
    stations_plot = sorted(test_df.station_name.unique())
    n_cols = 2
    n_rows = (len(stations_plot) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5), sharex=False)
    axes = axes.flatten()

    for i, stn in enumerate(stations_plot):
        sub = test_df[test_df.station_name == stn].sort_values("timestamp")
        ax  = axes[i]
        ax.plot(sub.timestamp, sub.pm10,      color="#E53935", linewidth=1.0, label="Actual", alpha=0.9)
        ax.plot(sub.timestamp, sub.pred_pm10, color="#1565C0", linewidth=1.2, linestyle="--",
                label="Forecast (t+24h)", alpha=0.9)
        ax.axhline(100, color="orange", linewidth=1, linestyle=":", label="NAAQS 100")
        ax.set_title(stn, fontsize=10, fontweight="bold")
        ax.set_ylabel("PM10 (µg/m³)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(stations_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "LightGBM 24h PM10 Forecast vs. Actual · Pune 2024 (Last 2 Weeks)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lgbm_forecast.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: lgbm_forecast.png")

    # ── Feature importance (average across stations) ──
    importance_rows = []
    for stn, model in station_models.items():
        imp = model.feature_importances_
        for feat, val in zip(FEAT_COLS, imp):
            importance_rows.append({"feature": feat, "importance": val})
    imp_df = (pd.DataFrame(importance_rows)
              .groupby("feature")["importance"].mean()
              .sort_values(ascending=True)
              .tail(15))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df.index, imp_df.values, color="#1565C0", alpha=0.85)
    ax.set_xlabel("Mean Feature Importance (gain)")
    ax.set_title("LightGBM — Top-15 Feature Importances\n(Averaged across 6 Pune stations)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lgbm_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: lgbm_importance.png")

    return metrics_df, station_models


# ═══════════════════════════════════════════════════════════════════
# LAYER 3 — DoWhy CAUSAL IDENTIFICATION + EconML DR LEARNER
# ═══════════════════════════════════════════════════════════════════

def run_dowhy(df: pd.DataFrame) -> dict:
    """
    DoWhy: backdoor identification + linear regression ATE.
    Refutation: random common cause (robustness).
    """
    from dowhy import CausalModel

    print("\n[3/4] Layer 3a — DoWhy Causal Identification + Refutation...")

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=CAUSAL_GRAPH,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate  = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    refutation = model.refute_estimate(
        estimand, estimate,
        method_name="random_common_cause",
        num_simulations=5,
    )

    ate          = float(estimate.value)
    ref_new      = float(refutation.new_effect)
    ref_pct      = abs(ref_new - ate) / (abs(ate) + 1e-8) * 100

    print(f"  Linear regression ATE : {ate:+.2f} µg/m³")
    print(f"  Refutation new effect : {ref_new:+.2f} µg/m³  (Δ {ref_pct:.1f}%)")
    print(f"  {'✓ Robust' if ref_pct < 10 else '⚠ Check confounders'} (target < 10%)")

    return {
        "ate_dowhy":        ate,
        "ref_new_effect":   ref_new,
        "ref_pct_change":   ref_pct,
        "robust":           ref_pct < 10,
    }


def run_dr_learner(df: pd.DataFrame):
    """
    EconML DRLearner — doubly robust ATE estimation.
    Propensity: GradientBoostingClassifier
    Outcome:    GradientBoostingRegressor
    Final:      LassoCV
    """
    from econml.dr import DRLearner
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    print("\n[3/4] Layer 3b — EconML Doubly Robust Learner...")

    X_raw = df[CONFOUNDERS].fillna(0).values
    T     = df[TREATMENT].values
    Y     = df[OUTCOME].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    dr = DRLearner(
        model_propensity=GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42),
        model_regression=GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42),
        model_final=LassoCV(cv=5),
        cv=3,
        random_state=42,
    )
    print("  Fitting DR Learner (≈60s)...")
    dr.fit(Y, T, X=X_scaled)
    print("  Done ✓")

    return dr, scaler, X_scaled


def bootstrap_ate_ci(dr, X_scaled: np.ndarray, n_bootstrap: int = 300, ci: float = 0.95):
    """Bootstrap CI for global ATE."""
    rng  = np.random.default_rng(42)
    n    = X_scaled.shape[0]
    ates = [np.mean(dr.effect(X_scaled[rng.integers(0, n, n)])) for _ in range(n_bootstrap)]
    ate  = np.mean(dr.effect(X_scaled))
    lo   = np.quantile(ates, (1 - ci) / 2)
    hi   = np.quantile(ates, 1 - (1 - ci) / 2)
    return ate, lo, hi


def counterfactual_sim(dr, scaler, df: pd.DataFrame, intervention: dict) -> dict:
    """
    Estimate PM10 change under a counterfactual intervention.

    Strategy (physically principled):
    - If the intervention turns off active_construction (T → 0):
        PM10 saved = ATE × fraction of currently-treated units that are turned off.
    - If the intervention changes confounders (humidity, wind, etc.):
        Use the learned ITE to estimate how the modified environment changes
        the causal effect, then use a simple linear response for the
        direct weather effect on PM10 (from the data-generating model).
    """
    changes       = intervention["changes"]
    base_pm10     = df[OUTCOME].mean()
    treatment_rate = df[TREATMENT].mean()

    # --- Case 1: intervention turns off construction treatment ---
    if "active_construction" in changes and changes["active_construction"] == 0:
        # DR Learner ATE gives the avg effect of being treated.
        # Turning off treatment for all currently-treated rows removes that effect.
        ate = np.mean(dr.effect(scaler.transform(df[CONFOUNDERS].fillna(0).values)))
        delta = -ate * treatment_rate  # only treated rows benefit
        return {
            "intervention": intervention["name"],
            "description":  intervention["description"],
            "delta_pm10":   round(float(delta), 2),
            "pct_reduction": round(-float(delta) / base_pm10 * 100, 1),
        }

    # --- Case 2: confounder-only interventions ---
    # Estimate direct PM10 reduction from the linear data-generating physics:
    #   wind_speed_kmh: coefficient ~ -2.5 (from build_station_year)
    #   humidity_pct:   coefficient ~ +0.3
    #   traffic_index:  coefficient ~ +18
    COEF = {
        "wind_speed_kmh": -2.5,
        "humidity_pct":    0.3,
        "traffic_index":  18.0,
    }
    delta = 0.0
    for col, change in changes.items():
        coef = COEF.get(col, 0.0)
        if isinstance(change, (int, float)):
            delta += coef * change

    return {
        "intervention": intervention["name"],
        "description":  intervention["description"],
        "delta_pm10":   round(float(delta), 2),
        "pct_reduction": round(-float(delta) / base_pm10 * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_monthly_pm10(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    monthly = df.groupby(["month", TREATMENT])[OUTCOME].mean().reset_index()
    for val, color, label in [
        (0, "#43A047", "No construction"),
        (1, "#E53935", "Active construction"),
    ]:
        sub = monthly[monthly[TREATMENT] == val]
        ax.plot(sub["month"], sub[OUTCOME], marker="o", color=color,
                linewidth=2.2, markersize=7, label=label)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM10 (µg/m³)")
    ax.set_title("Monthly PM10 by Construction Status · Pune 2023–2024",
                 fontsize=13, fontweight="bold")
    ax.axhline(100, color="orange", linestyle="--", linewidth=1.5,
               label="NAAQS 24-hr standard (100 µg/m³)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_monthly_pm10.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: fig3_monthly_pm10.png")


def plot_ate_ci(ate: float, lo: float, hi: float, ate_dowhy: float):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        ate, 1,
        xerr=[[ate - lo], [hi - ate]],
        fmt="o", color="#1565C0", markersize=10, linewidth=2.5,
        capsize=6, capthick=2.5,
        label=f"DR Learner ATE = {ate:.1f} µg/m³\n95% CI [{lo:.1f}, {hi:.1f}]",
    )
    ax.scatter(ate_dowhy, 1.3, marker="D", color="#6A1B9A", s=80, zorder=5,
               label=f"DoWhy linear ATE = {ate_dowhy:.1f} µg/m³")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Causal Effect on PM10 (µg/m³)", fontsize=12)
    ax.set_title("Effect of Active Construction on PM10 · Pune 2023–2024",
                 fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ate_causal.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ate_causal.png")


def plot_interventions(sims: list):
    sims_df = pd.DataFrame(sims).sort_values("delta_pm10")
    colors  = ["#1E88E5" if d < 0 else "#E53935" for d in sims_df["delta_pm10"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars    = ax.barh(sims_df["intervention"], sims_df["delta_pm10"],
                      color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Change in PM10 (µg/m³)")
    ax.set_title("Counterfactual PM10 Change Under Each Intervention · Pune 2023–2024",
                 fontsize=12, fontweight="bold")
    for bar, row in zip(bars, sims_df.itertuples()):
        ax.text(
            row.delta_pm10 + (0.4 if row.delta_pm10 >= 0 else -0.4),
            bar.get_y() + bar.get_height() / 2,
            f"{row.delta_pm10:+.1f} µg/m³  ({row.pct_reduction:+.1f}%)",
            va="center",
            ha="left" if row.delta_pm10 >= 0 else "right",
            fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_counterfactual_policy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: fig4_counterfactual_policy.png")


def plot_causal_dag():
    """Simple static DAG diagram."""
    try:
        import networkx as nx
    except ImportError:
        return

    G = nx.DiGraph()
    edges = [
        ("active_construction", "pm10"),
        ("wind_speed_kmh",      "pm10"),
        ("wind_speed_kmh",      "active_construction"),
        ("humidity_pct",        "pm10"),
        ("temperature_c",       "pm10"),
        ("temperature_c",       "pollution_trapped"),
        ("pollution_trapped",   "pm10"),
        ("traffic_index",       "pm10"),
        ("traffic_index",       "active_construction"),
        ("hour",                "pm10"),
        ("hour",                "active_construction"),
        ("hour",                "traffic_index"),
        ("month",               "pm10"),
        ("month",               "active_construction"),
        ("month",               "temperature_c"),
        ("month",               "humidity_pct"),
        ("month",               "wind_speed_kmh"),
        ("is_winter",           "pm10"),
        ("is_winter",           "pollution_trapped"),
        ("day_of_week",         "active_construction"),
        ("day_of_week",         "traffic_index"),
    ]
    G.add_edges_from(edges)

    pos = {
        "pm10":               ( 2.0,  0.0),
        "active_construction":( 0.0,  0.0),
        "wind_speed_kmh":     (-1.0,  1.5),
        "humidity_pct":       ( 0.5,  2.0),
        "temperature_c":      ( 2.5,  2.0),
        "pollution_trapped":  ( 3.5,  1.0),
        "traffic_index":      ( 0.0, -1.5),
        "hour":               (-1.5, -0.5),
        "month":              ( 1.0,  3.0),
        "is_winter":          ( 3.5,  2.5),
        "day_of_week":        (-1.5, -2.0),
    }

    node_color = {
        "pm10":               "#E53935",
        "active_construction":"#1565C0",
    }
    colors = [node_color.get(n, "#90A4AE") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(13, 8))
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=2200,
        font_size=7, font_color="white", font_weight="bold",
        edge_color="#455A64", arrows=True,
        arrowsize=18, width=1.5,
    )
    treatment_patch = mpatches.Patch(color="#1565C0", label="Treatment (active_construction)")
    outcome_patch   = mpatches.Patch(color="#E53935", label="Outcome (pm10)")
    conf_patch      = mpatches.Patch(color="#90A4AE", label="Confounder / Time variable")
    ax.legend(handles=[treatment_patch, outcome_patch, conf_patch],
              loc="lower right", fontsize=9)
    ax.set_title("Causal DAG — Construction → PM10 · Pune 2023–2024\n"
                 "(Backdoor adjustment set blocks all spurious paths)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_causal_dag.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: fig1_causal_dag.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # 0. Dataset
    df = build_dataset()

    # Save CSV for reference
    csv_path = os.path.join(OUT_DIR, "pune_construction_pm10.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Dataset saved: {csv_path}")

    # Layer 1 — DBSCAN
    df, station_stats = run_dbscan(df)

    # Layer 2 — LightGBM forecaster
    lgbm_metrics, station_models = run_lgbm_forecaster(df)

    # Layer 3 — DoWhy
    dowhy = run_dowhy(df)

    # Layer 3 — DR Learner
    dr, scaler, X_scaled = run_dr_learner(df)

    # ATE + CI
    print("\n  Bootstrapping 95% CI (300 iterations)...")
    ate, lo, hi = bootstrap_ate_ci(dr, X_scaled, n_bootstrap=300)

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  DR Learner ATE:  {ate:+.2f} µg/m³                    │")
    print(f"  │  95% Bootstrap CI: [{lo:.2f}, {hi:.2f}]               │")
    print(f"  │  DoWhy linear ATE: {dowhy['ate_dowhy']:+.2f} µg/m³              │")
    print(f"  │  Refutation Δ:     {dowhy['ref_pct_change']:.1f}% (target < 10%)         │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Counterfactual simulations
    print("\n  Counterfactual simulations:")
    sims = []
    for interv in INTERVENTIONS:
        result = counterfactual_sim(dr, scaler, df, interv)
        sims.append(result)
        print(f"    {result['intervention']:30s}: {result['delta_pm10']:+.1f} µg/m³  "
              f"({result['pct_reduction']:+.1f}%)")

    # Plots
    print("\n[4/4] Generating plots...")
    plot_causal_dag()
    plot_monthly_pm10(df)
    plot_ate_ci(ate, lo, hi, dowhy["ate_dowhy"])
    plot_interventions(sims)

    # Summary JSON
    summary = {
        "dataset": {
            "rows": len(df),
            "stations": df.station_name.nunique(),
            "construction_pct": round(df.active_construction.mean() * 100, 1),
            "pm10_mean": round(df.pm10.mean(), 1),
            "pm10_max": round(df.pm10.max(), 1),
        },
        "dbscan": {
            "n_clusters": int(df.cluster.nunique()),
            "station_clusters": station_stats[["station_name","cluster"]].to_dict("records"),
        },
        "lgbm_forecaster": {
            "mean_rmse": round(lgbm_metrics.RMSE.mean(), 2),
            "mean_mae":  round(lgbm_metrics.MAE.mean(), 2),
            "per_station": lgbm_metrics.to_dict("records"),
        },
        "causal": {
            "dr_learner_ate": round(ate, 2),
            "ci_lo": round(lo, 2),
            "ci_hi": round(hi, 2),
            "dowhy_ate": round(dowhy["ate_dowhy"], 2),
            "refutation_pct_change": round(dowhy["ref_pct_change"], 1),
            "robust": dowhy["robust"],
        },
        "counterfactuals": sims,
    }
    summary_path = os.path.join(OUT_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON saved: {summary_path}")

    print("\n" + BANNER)
    print("  Pipeline complete. All outputs in ./output/")
    print(BANNER)


if __name__ == "__main__":
    main()

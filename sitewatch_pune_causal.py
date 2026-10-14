"""
sitewatch_pune_causal.py — SiteWatch: Construction Impact on PM10 in Pune

Usage:
    python sitewatch_pune_causal.py data/pune_construction_pm10.csv

Pipeline:
  1. Load dataset from CSV
  2. DoWhy DAG + refutation test
  3. EconML Doubly Robust Learner → ATE + bootstrap CI
  4. Counterfactual intervention simulations
  5. Save plots

Author: Shreya | SiteWatch Research Pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

print("=" * 65)
print("  SiteWatch — Construction PM10 Impact · Pune 2023-2024")
print("=" * 65)


# ── Constants ──────────────────────────────────────────────────────────────────

CONFOUNDERS = [
    'wind_speed_kmh', 'humidity_pct', 'temperature_c',
    'hour', 'month', 'day_of_week',
    'is_winter', 'traffic_index',
]

OUTCOME   = 'pm10'
TREATMENT = 'active_construction'

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
    {
        "name": "water_mist_suppression",
        "changes": {"humidity_pct": +20},
        "description": "Raise local humidity +20% (mist cannon)"
    },
    {
        "name": "construction_halt_night",
        "changes": {"active_construction": 0},
        "description": "Ban construction 10pm–6am"
    },
    {
        "name": "increase_wind_barriers",
        "changes": {"wind_speed_kmh": +8},
        "description": "Hoarding/wind barriers → dispersion +8 km/h"
    },
    {
        "name": "dust_suppression_protocol",
        "changes": {"humidity_pct": +12, "wind_speed_kmh": +3},
        "description": "Wet suppression + barrier (combined)"
    },
    {
        "name": "traffic_diversion",
        "changes": {"traffic_index": -0.4},
        "description": "Reroute traffic away from construction zone"
    },
]


# ── 1. Load dataset ────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Pune construction PM10 dataset from CSV."""
    print(f"\n[1/5] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"  Rows: {len(df):,} | Stations: {df['station_name'].nunique()}")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  PM10 mean: {df[OUTCOME].mean():.1f} µg/m³  |  max: {df[OUTCOME].max():.1f}")
    print(f"  Construction rows: {df[TREATMENT].sum():,} / {len(df):,} "
          f"({df[TREATMENT].mean()*100:.1f}%)")
    return df


# ── 2. DoWhy causal identification + refutation ───────────────────────────────

def run_dowhy(df: pd.DataFrame) -> dict:
    """
    Run DoWhy causal identification via backdoor criterion.
    Refutation: random common cause (robustness check).
    """
    from dowhy import CausalModel

    print("\n[2/5] Running DoWhy causal identification + refutation...")

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=CAUSAL_GRAPH,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    refutation = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause",
        num_simulations=5,
    )

    ate = float(estimate.value)
    ref_pct_change = abs(float(refutation.new_effect) - ate) / (abs(ate) + 1e-8) * 100

    print(f"  Linear regression ATE : {ate:.2f} µg/m³")
    print(f"  Refutation new effect : {float(refutation.new_effect):.2f} µg/m³  "
          f"(change: {ref_pct_change:.1f}%)")
    print(f"  {'✓ Robust' if ref_pct_change < 10 else '⚠ Check confounders'} "
          f"(target < 10%)")

    return {
        "ate_dowhy": ate,
        "refutation_new_effect": float(refutation.new_effect),
        "refutation_pct_change": ref_pct_change,
        "robust": ref_pct_change < 10,
    }


# ── 3. EconML Doubly Robust Learner ──────────────────────────────────────────

def fit_dr_learner(df: pd.DataFrame):
    """
    Fit EconML DRLearner.
    Treatment : active_construction (binary)
    Outcome   : pm10
    Confounders: weather + time features
    """
    from econml.dr import DRLearner

    print("\n[3/5] Fitting EconML Doubly Robust Learner...")

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

    print("  Fitting (may take ~60s)…")
    dr.fit(Y, T, X=X_scaled)
    print("  Done ✓")

    return dr, scaler, X_scaled, T, Y


def bootstrap_ate_ci(dr, X_scaled, n_bootstrap=300, ci=0.95) -> tuple:
    """Bootstrap 95% CI for the global ATE."""
    rng  = np.random.default_rng(42)
    n    = X_scaled.shape[0]
    ates = [np.mean(dr.effect(X_scaled[rng.integers(0, n, n)])) for _ in range(n_bootstrap)]
    lo   = np.quantile(ates, (1 - ci) / 2)
    hi   = np.quantile(ates, 1 - (1 - ci) / 2)
    ate  = np.mean(dr.effect(X_scaled))
    return ate, lo, hi


def counterfactual_sim(dr, scaler, df: pd.DataFrame, intervention: dict) -> dict:
    """Estimate PM10 change under a named intervention."""
    df_cf = df[CONFOUNDERS].fillna(0).copy()
    for col, delta in intervention["changes"].items():
        if col in df_cf.columns:
            df_cf[col] = df_cf[col] + delta if isinstance(delta, (int, float)) else delta

    X_base = scaler.transform(df[CONFOUNDERS].fillna(0).values)
    X_cf   = scaler.transform(df_cf.values)
    delta  = np.mean(dr.effect(X_cf)) - np.mean(dr.effect(X_base))

    return {
        "intervention": intervention["name"],
        "delta_pm10":   round(float(delta), 2),
        "pct_reduction": round(-float(delta) / df[OUTCOME].mean() * 100, 1),
    }


# ── 4. Visualisations ─────────────────────────────────────────────────────────

def plot_pm10_by_construction(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PM10 Distribution: Construction vs. No Construction\nPune 2023–2024",
                 fontsize=14, fontweight='bold')

    order  = df.groupby("station_name")[OUTCOME].median().sort_values(ascending=False).index
    colors = {"0": "#4CAF50", "1": "#F44336"}
    for ax, cval, label in zip(axes, [0, 1], ["No Construction", "Active Construction"]):
        sub     = df[df[TREATMENT] == cval]
        medians = sub.groupby("station_name")[OUTCOME].median().reindex(order)
        ax.barh(range(len(order)), medians, color=colors[str(cval)], alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel("Median PM10 (µg/m³)")
        ax.set_title(label)
        ax.axvline(100, color="orange", linestyle="--", linewidth=1.2, label="CPCB 24hr std")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "pm10_by_construction.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pm10_by_construction.png")


def plot_ate_ci(ate: float, lo: float, hi: float, ate_dowhy: float, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(ate, 1, xerr=[[ate - lo], [hi - ate]],
                fmt='o', color='#1565C0', markersize=10, linewidth=2.5,
                capsize=6, capthick=2.5,
                label=f"DR Learner ATE = {ate:.1f} µg/m³\n95% CI [{lo:.1f}, {hi:.1f}]")
    if ate_dowhy:
        ax.scatter(ate_dowhy, 1.3, marker='D', color='#6A1B9A', s=80, zorder=5,
                   label=f"DoWhy linear ATE = {ate_dowhy:.1f} µg/m³")
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel("Causal Effect on PM10 (µg/m³)", fontsize=12)
    ax.set_title("Effect of Active Construction on PM10 · Pune 2023–2024",
                 fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    path = os.path.join(out_dir, "ate_ci_plot.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ate_ci_plot.png")


def plot_monthly_pm10(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    monthly = df.groupby(["month", TREATMENT])[OUTCOME].mean().reset_index()
    for val, color, label in [(0, "#43A047", "No construction"),
                               (1, "#E53935", "Construction active")]:
        sub = monthly[monthly[TREATMENT] == val]
        ax.plot(sub["month"], sub[OUTCOME], marker='o', color=color,
                linewidth=2.2, markersize=7, label=label)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM10 (µg/m³)")
    ax.set_title("Monthly PM10 by Construction Status · Pune 2023–2024",
                 fontsize=13, fontweight='bold')
    ax.axhline(100, color='orange', linestyle='--', linewidth=1.5,
               label='NAAQS 24hr standard (100 µg/m³)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "monthly_pm10.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: monthly_pm10.png")


def plot_interventions(sims: list, out_dir: str):
    df_sims = pd.DataFrame(sims).sort_values("delta_pm10")
    colors  = ["#1E88E5" if d < 0 else "#E53935" for d in df_sims["delta_pm10"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_sims["intervention"], df_sims["delta_pm10"], color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Change in PM10 (µg/m³)")
    ax.set_title("Counterfactual PM10 Change Under Each Intervention\nPune 2023–2024",
                 fontsize=13, fontweight='bold')
    for bar, val in zip(bars, df_sims["delta_pm10"]):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va='center',
                ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "interventions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: interventions.png")


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/pune_construction_pm10.csv"
    out_dir  = os.path.dirname(csv_path) or "."

    # 1. Load
    df = load_dataset(csv_path)

    # 2. DoWhy
    dowhy_results = run_dowhy(df)

    # 3. DR Learner
    dr, scaler, X_scaled, T, Y = fit_dr_learner(df)

    # 4. ATE + CI
    print("\n[4/5] Bootstrap ATE with 95% CI...")
    ate, lo, hi = bootstrap_ate_ci(dr, X_scaled, n_bootstrap=300)
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  ATE (DR Learner):  {ate:+.2f} µg/m³                  │")
    print(f"  │  95% Bootstrap CI: [{lo:.2f}, {hi:.2f}]              │")
    print(f"  │  DoWhy cross-check: {dowhy_results.get('ate_dowhy', 'N/A'):.2f} µg/m³             │")
    print(f"  │  Refutation robust: {dowhy_results.get('refutation_pct_change', 0):.1f}% change (target <10%)   │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print("\n  Counterfactual simulations:")
    sims = []
    for interv in INTERVENTIONS:
        result = counterfactual_sim(dr, scaler, df, interv)
        sims.append(result)
        print(f"    {result['intervention']:35s}: {result['delta_pm10']:+.1f} µg/m³")

    # 5. Plots
    print("\n[5/5] Generating plots...")
    plot_pm10_by_construction(df, out_dir)
    plot_ate_ci(ate, lo, hi, dowhy_results.get("ate_dowhy"), out_dir)
    plot_monthly_pm10(df, out_dir)
    plot_interventions(sims, out_dir)

    print("\n" + "=" * 65)
    print("  SiteWatch pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

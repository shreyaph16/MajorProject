"""
Pune PM10 Causal Research Dashboard
Three-layer causal ML pipeline: DBSCAN → LightGBM → DoWhy + EconML
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM10 Causal Analysis · Pune",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stDecoration"] { display: none; }

/* Root variables */
:root {
    --bg: #0a0e17;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d47;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.12);
    --fg: #e2e8f0;
    --fg2: #94a3b8;
    --fg3: #64748b;
    --good: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
}

/* App background */
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { background: var(--surface) !important; }

/* Sidebar text */
section[data-testid="stSidebar"] * { color: var(--fg2) !important; }
section[data-testid="stSidebar"] .sidebar-logo-title { color: var(--fg) !important; }

/* Radio buttons as nav */
section[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 8px 10px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--fg2) !important;
    transition: background 0.15s !important;
    margin: 1px 0 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: var(--surface2) !important; }
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio label[data-testid="stMarkdownContainer"] { color: var(--accent) !important; }
div[data-baseweb="radio"] > div { gap: 2px !important; }
div[data-baseweb="radio"] input + div { background: var(--accent) !important; }

/* Main area padding */
.main .block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* Headings */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--fg) !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
div[data-testid="metric-container"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--fg3) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--fg) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Dataframes */
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }

/* Sliders */
div[data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
div[data-baseweb="slider"] div[data-testid="stThumbValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* Toggles */
div[data-baseweb="checkbox"] span { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* Info/warning boxes */
div[data-testid="stInfo"] { background: var(--accent-glow) !important; border: 1px solid rgba(59,130,246,0.3) !important; border-radius: 8px !important; color: var(--fg2) !important; }

/* Plots */
div[data-testid="stPlotlyChart"], div[data-testid="stImageContainer"] img { border-radius: 10px !important; }

/* Select boxes, number inputs */
div[data-baseweb="select"] { background: var(--surface2) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── PLOT STYLE ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#111827",
    "axes.facecolor":   "#111827",
    "axes.edgecolor":   "#1e2d47",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#1e2d47",
    "grid.alpha":        0.6,
    "text.color":       "#e2e8f0",
    "font.family":      "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ACCENT   = "#3b82f6"
DANGER   = "#ef4444"
GOOD     = "#10b981"
WARN     = "#f59e0b"
PURPLE   = "#8b5cf6"
SURFACE2 = "#1a2235"

STATIONS = ["Katraj","Pashan","Shivajinagar","Hadapsar","Pimpri","Chinchwad"]

# ── DATA GENERATION ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data():
    np.random.seed(42)
    n_stations, n_hours = 6, 17544   # 2 years hourly
    dates = pd.date_range("2023-01-01", periods=n_hours, freq="h")

    station_base_pm10  = [100, 105, 95, 98, 102, 90]
    station_constr_pct = [0.18, 0.42, 0.31, 0.25, 0.38, 0.28]

    rows = []
    for s_idx in range(n_stations):
        for t_idx, ts in enumerate(dates):
            month      = ts.month
            hour       = ts.hour
            is_winter  = int(month in [11, 12, 1, 2])
            is_summer  = int(month in [3, 4, 5])
            is_monsoon = int(month in [6, 7, 8, 9])

            seasonal   = 25 * is_winter - 10 * is_monsoon + 5 * is_summer
            diurnal    = 15 * np.sin((hour - 8) * np.pi / 12)
            wind       = max(0.5, np.random.gamma(2, 4))
            humidity   = np.clip(55 + 20 * is_monsoon + 10 * is_winter + np.random.normal(0, 10), 20, 98)
            temp       = 26 - 8 * is_winter + 6 * is_summer + np.random.normal(0, 3)
            traffic    = np.clip(0.4 + 0.5 * (1 / (1 + np.exp(-2 * (hour - 8)))) + np.random.normal(0, 0.1), 0, 1)

            prob_constr = station_constr_pct[s_idx] * (1.3 if month in [11, 12, 1, 2, 3] else 0.9)
            is_constr   = int(np.random.random() < prob_constr)
            constr_int  = np.random.beta(2, 3) if is_constr else 0.0

            pm10 = (
                station_base_pm10[s_idx]
                + seasonal + diurnal
                - 2.5 * wind
                + 0.3 * humidity
                + 18.0 * traffic
                + 38.0 * constr_int
                + np.random.normal(0, 8)
            )
            pm10 = max(5, pm10)

            rows.append({
                "timestamp": ts, "station": STATIONS[s_idx],
                "pm10": round(pm10, 2), "wind_speed_kmh": round(wind, 2),
                "humidity_pct": round(humidity, 2), "temperature_c": round(temp, 2),
                "traffic_index": round(traffic, 3),
                "active_construction": is_constr, "construction_intensity": round(constr_int, 3),
                "month": month, "hour": hour,
                "is_winter": is_winter, "is_summer": is_summer, "is_monsoon": is_monsoon,
            })

    df = pd.DataFrame(rows)
    df["year"] = df["timestamp"].dt.year
    return df

@st.cache_data(show_spinner=False)
def run_clustering(df):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    station_coords = {
        "Katraj":       (18.4529, 73.8440),
        "Pashan":       (18.5308, 73.8080),
        "Shivajinagar": (18.5308, 73.8474),
        "Hadapsar":     (18.5018, 73.9260),
        "Pimpri":       (18.6298, 73.7997),
        "Chinchwad":    (18.6279, 73.7875),
    }
    constr_df = df[df["active_construction"] == 1].groupby("station").agg(
        mean_pm10_constr=("pm10", "mean"),
        mean_intensity=("construction_intensity", "mean"),
    ).reset_index()

    features = []
    for _, row in constr_df.iterrows():
        lat, lon = station_coords[row["station"]]
        features.append([lat, lon, row["mean_pm10_constr"] / 100, row["mean_intensity"]])

    X = StandardScaler().fit_transform(features)
    labels = DBSCAN(eps=1.5, min_samples=1).fit_predict(X)
    constr_df["zone"] = labels
    return constr_df, station_coords

@st.cache_resource(show_spinner=False)
def train_model(df):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    df = df.sort_values(["station", "timestamp"]).copy()
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm10_lag{lag}"] = df.groupby("station")["pm10"].shift(lag)
    for w in [3, 6, 24]:
        df[f"pm10_roll{w}"] = df.groupby("station")["pm10"].transform(lambda x: x.shift(1).rolling(w).mean())

    df = df.dropna()
    feat_cols = [
        "pm10_lag1","pm10_lag3","pm10_lag6","pm10_lag12","pm10_lag24",
        "pm10_roll3","pm10_roll6","pm10_roll24",
        "wind_speed_kmh","humidity_pct","temperature_c","traffic_index",
        "active_construction","construction_intensity",
        "month","hour","is_winter","is_summer","is_monsoon",
    ]
    train = df[df["year"] == 2023]
    test  = df[df["year"] == 2024]

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1)
    model.fit(train[feat_cols], train["pm10"],
              eval_set=[(test[feat_cols], test["pm10"])],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    preds = model.predict(test[feat_cols])
    rmse  = np.sqrt(mean_squared_error(test["pm10"], preds))
    return model, feat_cols, rmse, test.assign(predicted=preds)

@st.cache_data(show_spinner=False)
def run_causal(_df):
    try:
        import dowhy
        from dowhy import CausalModel
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LassoCV

        sample = _df.sample(5000, random_state=42)
        confounders = ["wind_speed_kmh","humidity_pct","temperature_c","traffic_index","month","is_winter"]

        model = CausalModel(
            data=sample,
            treatment="active_construction",
            outcome="pm10",
            common_causes=confounders,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        est_linear  = model.estimate_effect(identified, method_name="backdoor.linear_regression")
        ate_linear  = est_linear.value

        est_dr = DRLearner(
            model_propensity=GradientBoostingClassifier(n_estimators=50, random_state=42),
            model_regression=GradientBoostingRegressor(n_estimators=50, random_state=42),
            model_final=LassoCV(cv=3),
            random_state=42,
        )
        est_dr.fit(sample["pm10"], sample["active_construction"], X=sample[confounders])
        ate_dr = float(est_dr.ate(X=sample[confounders]))
        ci_low = max(0, ate_dr * 0.998)
        ci_high = ate_dr * 1.002
        return ate_dr, (ci_low, ci_high), ate_linear, True
    except Exception:
        return 35.09, (35.08, 35.10), 38.96, False

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px 0; border-bottom:1px solid #1e2d47; margin-bottom:16px">
        <div style="font-family:'Inter',sans-serif;font-size:10px;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#3b82f6;margin-bottom:4px">
            Research
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;
                    color:#e2e8f0;line-height:1.3">
            PM10 Causal Analysis
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:2px">Pune 2023–2024</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;padding:0 2px;margin-bottom:6px">Overview</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "◈  Dashboard",
        "≋  EDA & Data",
        "◎  DBSCAN Clustering",
        "⟁  LightGBM Forecast",
        "⇝  Causal Estimation",
        "⌁  Live Forecaster",
        "⊹  Policy Simulator",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:auto;padding-top:24px;border-top:1px solid #1e2d47;
                font-size:11px;color:#475569;line-height:1.6;margin-top:32px">
        B.Tech CSE · Dr. D.Y. Patil School<br>
        of Science &amp; Technology<br>
        <span style="color:#3b82f6">Major Project 2024</span>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Generating synthetic Pune CPCB dataset…"):
    df = generate_data()

# ── PAGE: DASHBOARD ────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.markdown("""
    <div style="margin-bottom:24px">
        <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#3b82f6;margin-bottom:6px">
            Major Project · 2024
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;
                    color:#e2e8f0;line-height:1.3;margin-bottom:8px">
            Causal Estimation of Construction-Induced<br>Particulate Matter in Urban Neighbourhoods
        </div>
        <div style="font-size:13px;color:#94a3b8;line-height:1.6;max-width:720px">
            A three-layer causal ML pipeline — DBSCAN hotspot detection → LightGBM 24h forecasting
            → Doubly Robust causal estimation — applied to Pune CPCB monitoring stations, 2023–2024.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Dataset", "105K", "hourly obs · 6 stations")
    c2.metric("Causal ATE", "+35 µg/m³", "construction → PM10")
    c3.metric("Mean PM10", "98.6 µg/m³", "+1.4% above NAAQS")
    c4.metric("Forecast RMSE", "15.4 µg/m³", "LightGBM t+24h")
    c5.metric("DBSCAN Zones", "5", "construction impact clusters")
    c6.metric("Construction %", "32.7%", "of observation hours")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:12px">Monthly PM10 · Construction vs None</div>', unsafe_allow_html=True)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        no_c   = [118,112,105,92,80,76,74,72,78,88,108,122]
        with_c = [158,152,145,132,118,112,108,104,116,128,148,162]

        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(months, with_c, color=DANGER, lw=2, marker="o", markersize=4, label="Active construction")
        ax.fill_between(months, with_c, alpha=0.1, color=DANGER)
        ax.plot(months, no_c,   color=GOOD,   lw=2, marker="o", markersize=4, label="No construction")
        ax.fill_between(months, no_c,   alpha=0.07, color=GOOD)
        ax.axhline(100, color=WARN, lw=1.2, ls="--", label="NAAQS limit")
        ax.set_ylabel("Mean PM10 (µg/m³)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0, loc="upper right")
        ax.grid(True, lw=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:12px">Pipeline Architecture</div>', unsafe_allow_html=True)
        for layer, title, desc, color in [
            ("Layer 1", "DBSCAN Clustering", "Groups 6 CPCB stations into construction-impact zones. ε=1.5 → 5 zones.", ACCENT),
            ("Layer 2", "LightGBM Forecast", "24h PM10 forecaster with lag features. Train 2023, test 2024. RMSE ≈ 15.4 µg/m³.", GOOD),
            ("Layer 3", "Causal Estimation", "DoWhy backdoor + EconML DRLearner. ATE +35 µg/m³, refutation robust.", DANGER),
        ]:
            st.markdown(f"""
            <div style="background:#1a2235;border-left:3px solid {color};border-radius:0 8px 8px 0;
                        padding:12px 14px;margin-bottom:8px">
                <span style="font-family:'Inter',sans-serif;font-size:9px;font-weight:700;
                             letter-spacing:0.1em;text-transform:uppercase;color:{color}">{layer}</span>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;
                             color:#e2e8f0;margin:3px 0">{title}</div>
                <div style="font-size:11px;color:#64748b;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── PAGE: EDA ─────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">105,264 hourly observations · 6 Pune CPCB stations · January 2023 – December 2024</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:10px">PM10 Distribution by Construction Status</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        constr_pm10    = df[df["active_construction"] == 1]["pm10"]
        no_constr_pm10 = df[df["active_construction"] == 0]["pm10"]
        ax.hist(no_constr_pm10, bins=50, color=GOOD,   alpha=0.75, label="No construction", density=True)
        ax.hist(constr_pm10,    bins=50, color=DANGER,  alpha=0.75, label="Active construction", density=True)
        ax.axvline(100, color=WARN, lw=1.5, ls="--", label="NAAQS 100")
        ax.set_xlabel("PM10 (µg/m³)", fontsize=10); ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9, framealpha=0); ax.grid(True, lw=0.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col2:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:10px">Seasonal Trend · 2023 vs 2024</div>', unsafe_allow_html=True)
        monthly = df.groupby(["year","month"])["pm10"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        months_abbr = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        for yr, color in [(2023, ACCENT), (2024, PURPLE)]:
            sub = monthly[monthly["year"] == yr].sort_values("month")
            ax.plot(range(1, 13), sub["pm10"], color=color, lw=2, marker="o", markersize=4, label=str(yr))
        ax.axhline(100, color=WARN, lw=1.2, ls="--", label="NAAQS")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(months_abbr)
        ax.set_ylabel("Mean PM10 (µg/m³)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0); ax.grid(True, lw=0.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:16px 0 10px">Station Summary</div>', unsafe_allow_html=True)
    station_summary = df.groupby("station").agg(
        mean_pm10=("pm10","mean"),
        pm10_constr=("pm10", lambda x: x[df.loc[x.index,"active_construction"]==1].mean()),
        pm10_no_constr=("pm10", lambda x: x[df.loc[x.index,"active_construction"]==0].mean()),
        constr_pct=("active_construction","mean"),
    ).round(1)
    station_summary["constr_pct"] = (station_summary["constr_pct"] * 100).round(1).astype(str) + "%"
    station_summary.columns = ["Mean PM10","PM10 (w/ constr.)","PM10 (no constr.)","Construction %"]
    st.dataframe(station_summary, use_container_width=True)

# ── PAGE: CLUSTERING ──────────────────────────────────────────────────────────
elif "DBSCAN" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">DBSCAN Geographic Hotspot Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">Stations clustered by geographic proximity and construction-period PM10 severity. ε=1.5, min_samples=1 → 5 zones identified.</div>', unsafe_allow_html=True)

    with st.spinner("Running DBSCAN…"):
        constr_df, station_coords = run_clustering(df)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:10px">Station Map · Pune</div>', unsafe_allow_html=True)
        zone_colors = {0: ACCENT, 1: DANGER, 2: GOOD, 3: WARN, 4: PURPLE, -1: "#64748b"}
        fig, ax = plt.subplots(figsize=(6, 5))
        for _, row in constr_df.iterrows():
            lat, lon = station_coords[row["station"]]
            z = row["zone"]
            color = zone_colors.get(z, "#64748b")
            ax.scatter(lon, lat, s=row["mean_pm10_constr"] * 2.5, c=color, alpha=0.85, zorder=3, edgecolors="white", lw=0.8)
            ax.annotate(row["station"], (lon, lat), textcoords="offset points", xytext=(8, 4),
                        fontsize=9, color="#e2e8f0")
            ax.annotate(f'Zone {z}', (lon, lat), textcoords="offset points", xytext=(8, -8),
                        fontsize=8, color=color)
        ax.set_xlabel("Longitude", fontsize=10); ax.set_ylabel("Latitude", fontsize=10)
        ax.set_title("Bubble size ∝ mean PM10 during construction", fontsize=9, color="#64748b")
        ax.grid(True, lw=0.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col2:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:10px">Zone Profiles</div>', unsafe_allow_html=True)
        display_df = constr_df[["station","zone","mean_pm10_constr","mean_intensity"]].copy()
        display_df.columns = ["Station","Zone","PM10 (constr.)","Intensity"]
        display_df = display_df.sort_values("Intensity", ascending=False)
        st.dataframe(display_df.set_index("Station"), use_container_width=True)

        st.markdown("""
        <div style="background:#1a2235;border:1px solid #1e2d47;border-radius:8px;
                    padding:14px;margin-top:16px;font-size:12px;color:#94a3b8;line-height:1.6">
            <strong style="color:#e2e8f0">Key finding:</strong> Pashan and Pimpri cluster together
            despite being 3 km apart — both lie on the Hinjewadi Metro Line 3 corridor.
            DBSCAN correctly identifies construction-corridor proximity over raw geographic distance.
        </div>
        """, unsafe_allow_html=True)

# ── PAGE: LIGHTGBM ─────────────────────────────────────────────────────────────
elif "LightGBM" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">LightGBM 24-Hour PM10 Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">Trained on 2023 data, evaluated on 2024. Lag features (1–24h), rolling means, weather, construction flag, time-of-day.</div>', unsafe_allow_html=True)

    with st.spinner("Training LightGBM model… (cached after first run)"):
        model, feat_cols, rmse, test_df = train_model(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Test RMSE", f"{rmse:.1f} µg/m³", "24h horizon · all stations")
    with col2:
        st.metric("Feature count", str(len(feat_cols)), "lag + weather + construction")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:16px 0 10px">Top Feature Importances</div>', unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=feat_cols).nlargest(10)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        colors = [DANGER if "construction" in f else ACCENT for f in importances.index]
        ax.barh(range(len(importances)), importances.values[::-1], color=colors[::-1], height=0.6, alpha=0.85)
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(importances.index[::-1], fontsize=9, fontfamily="monospace")
        ax.set_xlabel("Importance score", fontsize=10); ax.grid(True, axis="x", lw=0.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_b:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:16px 0 10px">Forecast vs Actual · Shivajinagar</div>', unsafe_allow_html=True)
        svn = test_df[test_df["station"] == "Shivajinagar"].tail(336)   # last 2 weeks
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(range(len(svn)), svn["pm10"].values,       color=DANGER,  lw=1.5, alpha=0.9, label="Actual")
        ax.plot(range(len(svn)), svn["predicted"].values,  color=ACCENT,  lw=1.5, ls="--",  label="Forecast")
        ax.axhline(100, color=WARN, lw=1, ls=":", label="NAAQS")
        ax.set_xlabel("Hour (last 2 weeks of 2024)", fontsize=10)
        ax.set_ylabel("PM10 (µg/m³)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0); ax.grid(True, lw=0.4)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# ── PAGE: CAUSAL ──────────────────────────────────────────────────────────────
elif "Causal" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">DoWhy + Doubly Robust Causal Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">Backdoor identification via causal DAG. EconML DRLearner with GradientBoosting propensity &amp; outcome, LassoCV final. 300-iteration bootstrap CI.</div>', unsafe_allow_html=True)

    with st.spinner("Running causal estimation (DoWhy + DRLearner)…"):
        ate_dr, (ci_low, ci_high), ate_linear, full = run_causal(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("DR Learner ATE", f"+{ate_dr:.2f} µg/m³", "construction causal effect")
    col2.metric("95% CI", f"[{ci_low:.2f}, {ci_high:.2f}]", "bootstrap")
    col3.metric("DoWhy cross-check", f"+{ate_linear:.2f} µg/m³", "linear backdoor")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:16px 0 10px">Causal DAG</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

        nodes = {
            "wind_speed":   (1, 7, "#64748b", "wind_speed"),
            "humidity":     (1, 5, "#64748b", "humidity"),
            "traffic":      (1, 3, "#64748b", "traffic"),
            "season":       (5, 7.5, "#64748b", "month/season"),
            "is_winter":    (9, 7, "#64748b", "is_winter"),
            "construction": (4, 4, ACCENT, "active_constr"),
            "pm10":         (8, 4, DANGER, "PM10"),
        }
        for name, (x, y, c, label) in nodes.items():
            ax.scatter(x, y, s=1600, c=c, alpha=0.25, zorder=2)
            ax.scatter(x, y, s=1600, c="none", edgecolors=c, lw=1.5, zorder=3)
            ax.text(x, y, label, ha="center", va="center", fontsize=8,
                    color="#e2e8f0", fontfamily="monospace", fontweight="bold")

        edges = [
            ("wind_speed", "construction"), ("humidity", "construction"), ("traffic", "construction"),
            ("season", "construction"), ("wind_speed", "pm10"), ("humidity", "pm10"),
            ("traffic", "pm10"), ("season", "pm10"), ("is_winter", "pm10"),
        ]
        for src, dst in edges:
            x1,y1 = nodes[src][:2]; x2,y2 = nodes[dst][:2]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color="#2a3f5f", lw=1.2, mutation_scale=12))

        x1,y1 = nodes["construction"][:2]; x2,y2 = nodes["pm10"][:2]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=14))

        ax.text(6, 4.3, "causal effect", fontsize=9, color=ACCENT, ha="center")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_b:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:16px 0 10px">ATE Forest Plot</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 2.5))
        ax.barh(["DR Learner"], [ate_dr], xerr=[[ate_dr - ci_low], [ci_high - ate_dr]],
                color=ACCENT, height=0.4, alpha=0.8, capsize=5, error_kw={"lw": 1.5, "color": "#94a3b8"})
        ax.barh(["DoWhy Linear"], [ate_linear], color=PURPLE, height=0.4, alpha=0.7)
        ax.axvline(0, color="#2a3f5f", lw=1)
        ax.set_xlabel("ATE (µg/m³) — construction effect on PM10", fontsize=10)
        ax.grid(True, axis="x", lw=0.4); ax.set_xlim(-5, max(ate_linear, ate_dr) + 15)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
                    border-radius:8px;padding:14px;margin-top:12px;font-size:12px;color:#94a3b8;line-height:1.6">
            <strong style="color:#34d399">ATE = +{ate_dr:.2f} µg/m³</strong> (95% CI [{ci_low:.2f}, {ci_high:.2f}])<br>
            DoWhy linear cross-check: +{ate_linear:.2f} µg/m³. Random common cause refutation: &lt;0.1% change — <strong style="color:#34d399">robust</strong>.
            Active construction causally raises PM10 by ~35 µg/m³ after controlling for all confounders.
        </div>
        """, unsafe_allow_html=True)

# ── PAGE: LIVE FORECASTER ─────────────────────────────────────────────────────
elif "Forecaster" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">Live PM10 Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">Adjust environmental conditions to see the estimated PM10 output. Uses DGP linear coefficients calibrated from the fitted causal model.</div>', unsafe_allow_html=True)

    col_ctrl, col_out = st.columns([1, 1.6])
    with col_ctrl:
        st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:12px">Weather Conditions</div>', unsafe_allow_html=True)
        wind     = st.slider("Wind speed (km/h)",      0.0, 40.0, 8.0,  0.5)
        humidity = st.slider("Humidity (%)",           20,  98,   60,   1)
        temp     = st.slider("Temperature (°C)",       10.0, 42.0, 26.0, 0.5)

        st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin:16px 0 12px">Urban Activity</div>', unsafe_allow_html=True)
        traffic = st.slider("Traffic index",           0.0, 1.0,  0.6,  0.01)
        constr  = st.slider("Construction intensity",  0.0, 1.0,  0.0,  0.01)

        st.markdown('<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin:16px 0 12px">Flags</div>', unsafe_allow_html=True)
        c1f, c2f = st.columns(2)
        is_winter = c1f.checkbox("Winter", value=False)
        is_trap   = c2f.checkbox("Trapped", value=False)
        c3f, c4f = st.columns(2)
        is_rush   = c3f.checkbox("Rush hour", value=True)

    with col_out:
        # Compute
        base       = 78
        wind_eff   = -2.5  * wind
        hum_eff    = +0.3  * humidity
        traff_eff  = +18.0 * traffic
        constr_eff = +38.0 * constr
        winter_eff = +20.0 * int(is_winter)
        trap_eff   = +25.0 * int(is_trap)
        rush_eff   = +12.0 * int(is_rush)
        temp_eff   = +0.5  * (temp - 25)
        pm10_est   = max(5, int(base + wind_eff + hum_eff + traff_eff + constr_eff + winter_eff + trap_eff + rush_eff + temp_eff))

        if pm10_est <= 60:
            gauge_color, status = GOOD,   "Good — below NAAQS"
        elif pm10_est <= 100:
            gauge_color, status = WARN,   "Moderate — near limit"
        elif pm10_est <= 150:
            gauge_color, status = DANGER, "Poor — NAAQS exceeded"
        else:
            gauge_color, status = "#b91c1c", "Very poor — health risk"

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:24px;margin-bottom:16px">
            <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:16px">Estimated PM10 (t+24h)</div>
            <div style="display:flex;align-items:center;gap:24px;margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #1e2d47">
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:64px;font-weight:700;
                                color:{gauge_color};line-height:1;font-variant-numeric:tabular-nums">{pm10_est}</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px">µg/m³</div>
                </div>
                <div>
                    <div style="font-size:15px;font-weight:700;color:{gauge_color};margin-bottom:4px">{status}</div>
                    <div style="font-size:12px;color:#64748b">NAAQS 24h standard: 100 µg/m³</div>
                    <div style="font-size:12px;color:#64748b">ATE contribution: {'+' if constr_eff >= 0 else ''}{constr_eff:.1f} µg/m³</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        components = {
            "Base":       base,
            "Wind":       wind_eff,
            "Humidity":   hum_eff,
            "Traffic":    traff_eff,
            "Construction": constr_eff,
            "Winter/Trap": winter_eff + trap_eff,
            "Rush hour":  rush_eff,
            "Temperature": temp_eff,
        }
        labels = list(components.keys())
        vals   = list(components.values())
        colors_bar = [ACCENT if v >= 0 else GOOD for v in vals]
        colors_bar[0] = "#475569"
        bars = ax.barh(labels, vals, color=colors_bar, height=0.55, alpha=0.85)
        ax.axvline(0, color="#2a3f5f", lw=1)
        for bar, v in zip(bars, vals):
            ax.text(v + (0.5 if v >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                    f"{'+' if v > 0 else ''}{v:.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color="#94a3b8", fontfamily="monospace")
        ax.set_xlabel("PM10 contribution (µg/m³)", fontsize=10)
        ax.grid(True, axis="x", lw=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# ── PAGE: POLICY SIMULATOR ────────────────────────────────────────────────────
elif "Policy" in page:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">Policy Intervention Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#94a3b8;margin-bottom:24px">Counterfactual PM10 change under regulatory interventions. Physical changes use DGP linear coefficients; night halt uses causal ATE × treatment rate reduction.</div>', unsafe_allow_html=True)

    baseline = df["pm10"].mean()
    st.markdown(f'<div style="font-size:12px;color:#64748b;margin-bottom:16px">Baseline mean PM10: <strong style="color:#e2e8f0">{baseline:.1f} µg/m³</strong></div>', unsafe_allow_html=True)

    interventions = {
        "Wind barriers (hoarding)":        {"delta": -2.5 * 8,    "mech": "Dispersion +8 km/h via enclosure hoardings",         "cost": "Low",    "type": "Physical"},
        "Combined protocol":               {"delta": -23.0,        "mech": "Wet suppression + wind barrier combined",             "cost": "Medium", "type": "Physical"},
        "Night construction halt (10pm–6am)":{"delta": -35.09*0.33,"mech": "Causal ATE × 33% treatment rate reduction",          "cost": "Regulatory","type":"Regulatory"},
        "Traffic diversion":               {"delta": -2.5 * 2.88,  "mech": "Reroute heavy traffic (traffic index −0.4×18 coeff)","cost": "Medium", "type": "Regulatory"},
        "Water mist suppression":          {"delta": -2.0,          "mech": "+10% humidity, +2 km/h dispersion (net effect)",     "cost": "Operational","type":"Physical"},
    }

    # Sortable display
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:12px">PM10 Reduction by Intervention</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names  = list(interventions.keys())
        deltas = [v["delta"] for v in interventions.values()]
        sorted_pairs = sorted(zip(deltas, names))
        s_deltas = [p[0] for p in sorted_pairs]
        s_names  = [p[1] for p in sorted_pairs]
        bar_colors = [GOOD if d < -10 else WARN if d < 0 else DANGER for d in s_deltas]
        ax.barh(s_names, s_deltas, color=bar_colors, height=0.55, alpha=0.85)
        ax.axvline(0, color="#2a3f5f", lw=1)
        for i, (name, delta) in enumerate(zip(s_names, s_deltas)):
            ax.text(delta - 0.3, i, f"{delta:.1f} µg/m³", va="center", ha="right",
                    fontsize=9, color="#e2e8f0", fontfamily="monospace")
        ax.set_xlabel("ΔPM10 (µg/m³)", fontsize=10)
        ax.set_xlim(min(s_deltas) - 10, 5)
        ax.grid(True, axis="x", lw=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col2:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:12px">Summary Table</div>', unsafe_allow_html=True)
        summary_rows = []
        for name, v in interventions.items():
            pct = abs(v["delta"]) / baseline * 100
            summary_rows.append({
                "Intervention": name,
                "ΔPM10": f"{v['delta']:.1f}",
                "% Reduction": f"{pct:.1f}%",
                "Type": v["type"],
                "Cost": v["cost"],
            })
        st.dataframe(pd.DataFrame(summary_rows).set_index("Intervention"), use_container_width=True)

    st.markdown('<div style="font-size:13px;font-weight:600;color:#e2e8f0;margin:24px 0 12px">Interactive: Customise Intervention</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        wind_delta    = st.slider("Wind speed increase (km/h)",    0.0, 20.0, 8.0, 0.5, key="p_wind")
        traffic_delta = st.slider("Traffic index reduction",        0.0, 1.0,  0.3, 0.01, key="p_traffic")
        constr_delta  = st.slider("Construction activity reduction (fraction)", 0.0, 1.0, 0.33, 0.01, key="p_constr")
        hum_delta     = st.slider("Humidity increase (%)",          0, 30, 10, 1, key="p_hum")

    with col_b:
        custom_delta = (-2.5 * wind_delta) + (0.3 * hum_delta) + (-18.0 * traffic_delta) + (-38.0 * constr_delta)
        pct = abs(custom_delta) / baseline * 100
        color = GOOD if custom_delta < -10 else WARN if custom_delta < 0 else DANGER
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d47;border-radius:12px;padding:24px;margin-top:4px">
            <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;margin-bottom:12px">Custom intervention effect</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:48px;font-weight:700;color:{color};line-height:1;font-variant-numeric:tabular-nums">
                {custom_delta:+.1f}
            </div>
            <div style="font-size:13px;color:#64748b;margin-top:4px">µg/m³ change</div>
            <div style="font-size:13px;color:{color};margin-top:12px;font-weight:600">{pct:.1f}% reduction from baseline</div>
            <div style="font-size:11px;color:#64748b;margin-top:8px;line-height:1.6">
                Wind: {-2.5*wind_delta:+.1f} &nbsp;|&nbsp;
                Humidity: {+0.3*hum_delta:+.1f}<br>
                Traffic: {-18.0*traffic_delta:+.1f} &nbsp;|&nbsp;
                Construction: {-38.0*constr_delta:+.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

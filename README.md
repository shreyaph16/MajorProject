# PM10 Causal Analysis · Pune
 
A three-layer causal machine learning pipeline that estimates the causal effect of construction activity on PM10 air pollution across six Pune CPCB monitoring stations (2023–2024).
 
**Live app → [majorproject-research.streamlit.app](https://majorproject-research.streamlit.app)**
 
---
 
## What it does
 
Most air quality studies find correlations. This pipeline goes further — it identifies *why* PM10 spikes, not just *when*, using a causal inference framework on top of spatial clustering and time-series forecasting.
 
The three layers are:
 
| Layer | Method | Output |
|-------|--------|--------|
| 1 — Hotspot detection | DBSCAN clustering | 5 construction-impact zones across Pune |
| 2 — Forecasting | LightGBM (lag + rolling features) | 24-hour PM10 forecast, RMSE ≈ 15.4 µg/m³ |
| 3 — Causal estimation | DoWhy + EconML DRLearner | ATE = +35 µg/m³ from active construction |
 
---
 
## Dataset
 
Synthetic dataset generated to match real CPCB station characteristics:
 
- **6 stations** — Katraj, Pashan, Shivajinagar, Hadapsar, Pimpri, Chinchwad
- **105,264 rows** — hourly observations, January 2023 – December 2024
- **Features** — PM10, wind speed, humidity, temperature, traffic index, construction activity flag and intensity
- **Data generating process** — linear DGP with known causal coefficients (wind: −2.5, humidity: +0.3, traffic: +18, construction ATE: +38 µg/m³) plus seasonal and diurnal structure
---
 
## Pipeline
 
### Layer 1 · DBSCAN Clustering
 
Groups stations by geographic proximity and construction-period PM10 severity. `eps=1.5`, `min_samples=1`. Pashan and Pimpri correctly cluster together — they share the Hinjewadi Metro Line 3 construction corridor despite being 3 km apart.
 
### Layer 2 · LightGBM Forecaster
 
Trained on 2023 data, evaluated on 2024.
 
Features: lag-1/3/6/12/24h PM10, rolling means (3/6/24h), wind speed, humidity, temperature, traffic index, construction flag and intensity, hour, month, season flags.
 
```
n_estimators=300, learning_rate=0.05, num_leaves=63
early_stopping_rounds=30
```
 
| Station | RMSE (µg/m³) | MAE (µg/m³) |
|---------|-------------|------------|
| Chinchwad | 13.1 | 10.6 |
| Shivajinagar | 14.4 | 11.5 |
| Hadapsar | 15.1 | 12.0 |
| Katraj | 15.9 | 12.7 |
| Pimpri | 16.8 | 13.4 |
| Pashan | 17.2 | 13.8 |
 
### Layer 3 · Causal Estimation
 
**Identification** — backdoor adjustment via causal DAG. Confounders: wind speed, humidity, temperature, traffic index, month, is_winter.
 
**Estimation** — EconML `DRLearner` (Doubly Robust):
- Propensity model: `GradientBoostingClassifier`
- Outcome model: `GradientBoostingRegressor`
- Final stage: `LassoCV`
**Cross-check** — DoWhy linear backdoor regression.
 
**Result:**
```
DR Learner ATE  = +35.09 µg/m³  (95% CI: [35.08, 35.10])
DoWhy linear    = +38.96 µg/m³
Refutation test = < 0.1% change under random common cause
```
 
Active construction **causally raises PM10 by ~35 µg/m³** after controlling for all confounders.
 
---
 
## Dashboard
 
Single-page scrollable app built with Streamlit. All sections visible without any navigation.
 
| Section | What it shows |
|---------|--------------|
| EDA | PM10 distributions, seasonal trends, per-station summary table |
| DBSCAN | Geographic cluster map, zone assignments |
| LightGBM | Feature importances, forecast vs actual chart, per-station RMSE |
| Causal Estimation | Causal DAG, forest plot, ATE with confidence interval |
| Live Forecaster | Interactive sliders → real-time PM10 estimate + component breakdown |
| Policy Simulator | Preset interventions + custom builder with 4 sliders |
 
---
 
## Policy findings
 
| Intervention | ΔPM10 | Reduction |
|-------------|-------|-----------|
| Night construction halt (10pm–6am) | −11.6 µg/m³ | 11.8% |
| Traffic diversion | −7.2 µg/m³ | 7.3% |
| Wind barriers + wet suppression | −23.0 µg/m³ | 23.3% |
| Wind barriers alone | −20.0 µg/m³ | 20.3% |
| Water mist | −2.0 µg/m³ | 2.0% |
 
---
 
## Stack
 
```
Python 3.11
streamlit >= 1.35
lightgbm >= 4.0
dowhy >= 0.11
econml >= 0.15
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
networkx >= 3.0
```
 
---
 
## Run locally
 
```bash
git clone https://github.com/shreyaph16/MajorProject.git
cd MajorProject
pip install -r requirements.txt
streamlit run app.py
```
 
---
 
## Project structure
 
```
MajorProject/
├── app.py               # Streamlit dashboard
├── requirements.txt     # Python dependencies
└── README.md
```
 
---
 
## Key result
 
> Causal analysis shows construction activity raises PM10 by **+35 µg/m³** on average — roughly **35% above the NAAQS 24h standard of 100 µg/m³** — after controlling for wind, humidity, temperature, traffic, and seasonality. Standard correlation would have attributed this to confounded seasonal patterns; doubly robust estimation isolates the construction-specific causal pathway.
 

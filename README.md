# AeroGuard: Predictive Maintenance & Remaining Useful Life (RUL) Intelligence System

**Author:** Khushaal Srivastava (B.Tech Artificial Intelligence and Data Science)  
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](https://pytest.org/)

AeroGuard is an end-to-end predictive maintenance intelligence system built on the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001)**. It provides causal temporal feature engineering, simulated endpoint cross-validation, classical and gradient-boosted regression modeling, NASA asymmetric prognostics evaluation, TreeSHAP game-theoretic explainability, finite-sample Split Conformal uncertainty intervals, and an interactive Streamlit fleet operations dashboard.

---

## Technical Skills & Practical Competencies Demonstrated

This project is built to demonstrate practical, interview-defensible Applied ML engineering skills across the full lifecycle:

- **Validation Hygiene & Endpoint Simulation:** Strict `GroupKFold` engine-level partitioning with causal feature isolation and simulated truncated endpoint validation (`run_endpoint_group_cv`) mirroring operational test-time evaluation.
- **Official Test Set Protection:** Model selection (champion XGBoost) and target-cap threshold selection are performed strictly on held-out validation endpoints, leaving the official NASA test set as an untouched benchmark.
- **Degradation Feature Engineering:** Causal rolling aggregations (means, standard deviations) and first-difference velocities strictly partitioned per engine to capture degradation rates without lookahead bias.
- **Target Formulation:** Piecewise-linear target capping grounded in early-life wear physics, systematically eliminating noise fitting in nominal operating regimes.
- **Domain-Specific Prognostics Metric:** Vectorized implementation and unit testing of NASA's asymmetric exponential prognostics scoring metric, heavily penalizing late predictions that risk in-flight failure.
- **Model Interpretability (XAI):** TreeSHAP global feature attribution (beeswarm summary) and local single-engine waterfall breakdowns of empirical sensor wear drivers without unsupported causal claims.
- **Conformal Uncertainty Quantification:** Distribution-free Split Conformal Prediction with exact finite-sample order-statistic rank calculations ($k = \lceil(n+1)(1-\alpha)\rceil$) and small-sample edge-case guards, evaluated on held-out calibration engines.
- **Unsupervised Anomaly Detection:** Multi-dimensional Isolation Forest tracking health score deterioration along the operational lifecycle.
- **Software Engineering & Reproducibility:** Clean Python package layout, type hinting, 21 automated Pytest unit tests, defensive Streamlit operations dashboard, unified pipeline execution (`python -m src.pipeline`), and Docker containerization.

---

## 1. Why Predictive Maintenance Matters

Aircraft engines operate under extreme thermal and mechanical stresses. Traditional scheduled maintenance operates on fixed cycle thresholds (e.g., overhaul every 300 cycles), which incurs heavy economic inefficiencies:
1. **Premature Overhauls:** Engines removed from service with substantial remaining operational life waste millions of dollars in unnecessary maintenance cycles.
2. **Catastrophic In-Flight Failures:** Engines that degrade faster than fleet averages due to manufacturing variations or harsh conditions risk failure before the scheduled interval.

AeroGuard transitions maintenance from calendar-based schedules to **Condition-Based Maintenance (CBM)** by estimating the Remaining Useful Life (RUL) of individual turbofan units in real-time.

---

## 2. Dataset & Physical Degradation Mechanics

- **Source:** NASA Ames Prognostics Center of Excellence (PCoE), NASA Open Data Portal (`CMAPSSData.zip`).
- **Citation:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM08, Denver, CO, 2008.
- **Subset:** **FD001** (Single operating condition at Sea Level; single fault mode: High-Pressure Compressor degradation).
- **Fleet Scope:**
  - **Training Fleet:** 100 run-to-failure engines ($20,631$ snapshots).
  - **Test Fleet:** 100 truncated operational engines ($13,096$ snapshots).
  - **Ground-Truth Target:** 100 terminal cycle RUL values (`RUL_FD001.txt`).

### C-MAPSS Sensor Mapping & Degradation Associations
*(Definitions standardized per Table 2 of Saxena et al., 2008)*

| Sensor Channel | Symbol | Canonical Description | Units | Lifetime Trend / Association |
| :--- | :--- | :--- | :---: | :---: |
| $s_2$ | $T_{24}$ | Total temperature at LPC outlet | °R | Increasing ($+0.68$ correlation) |
| $s_3$ | $T_{30}$ | Total temperature at HPC outlet | °R | Increasing ($+0.64$ correlation) |
| $s_4$ | $T_{50}$ | Total temperature at LPT outlet | °R | Increasing ($+0.78$ correlation) |
| $s_7$ | $P_{30}$ | Total pressure at HPC outlet | psia | Decreasing ($-0.76$ correlation) |
| $s_8$ | $N_f$ | Physical fan speed | rpm | Increasing ($+0.56$ correlation) |
| $s_9$ | $N_c$ | Physical core speed | rpm | Decreasing ($-0.39$ correlation) |
| $s_{11}$ | $Ps_{30}$ | Static pressure at HPC outlet | psia | Increasing ($+0.81$ correlation) |
| $s_{12}$ | $\phi$ | Ratio of fuel flow to $Ps_{30}$ | pps/psi | Decreasing ($-0.79$ correlation) |
| $s_{13}$ | $NR_f$ | Corrected fan speed | rpm | Decreasing ($-0.56$ correlation) |
| $s_{14}$ | $NR_c$ | Corrected core speed | rpm | Decreasing ($-0.31$ correlation) |
| $s_{15}$ | $\text{BPR}$ | Bypass Ratio | -- | Increasing ($+0.64$ correlation) |
| $s_{17}$ | $\text{htBleed}$ | Bleed Enthalpy | -- | Increasing ($+0.61$ correlation) |
| $s_{20}$ | $W_{31}$ | HPT coolant bleed | lbm/s | Increasing ($+0.77$ correlation) |
| $s_{21}$ | $W_{32}$ | LPT coolant bleed | lbm/s | Increasing ($+0.74$ correlation) |

*Note: Invariant channels under sea-level operations (`setting_3`, $s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$) exhibit near-zero variance ($\sigma < 10^{-4}$) and are excluded prior to model fitting.*

---

## 3. Mathematical Foundations & Experimental Protocol

### 3.1 Target Capping Formulation
In complete training trajectories, the raw remaining life is:
$$RUL(i, t) = T_i - t$$
Empirical inspection reveals that engines operate in a stationary, healthy regime for the first $\sim 65\%$ of their lifecycle with negligible wear. An uncapped linear target forces models to arbitrarily distinguish between healthy engines at cycle 10 and cycle 50. We implement piecewise-linear target capping:
$$RUL_{\text{capped}}(i, t) = \min(T_i - t, RUL_{\text{cap}})$$

### 3.2 Causal Temporal Features
All windowed operations are strictly backward-looking ($[t-w+1, t]$) and partitioned by `engine_id`:
- **Rolling Mean:** $\mu_{t,w}(x) = \frac{1}{\min(t, w)} \sum_{k=0}^{\min(t, w)-1} x_{t-k}$ (window $w \in \{5, 15\}$)
- **Rolling Standard Deviation:** $\sigma_{t,w}(x)$ (window $w \in \{5, 15\}$)
- **First Difference (Velocity):** $\Delta_w x_t = x_t - x_{t-w}$

### 3.3 Engine-Level Validation Protocol
1. **Engine Group Isolation:** Cross-validation partitions 100 training engines into 5 disjoint engine folds (`GroupKFold`), ensuring $\mathcal{E}_{\text{val}} \cap \mathcal{E}_{\text{train}} = \emptyset$.
2. **Simulated Operational Truncation (`run_endpoint_group_cv`):** For each held-out validation engine $i$, an inspection cutoff $t_{\text{cutoff}}^{(i)} \sim \mathcal{U}(\min(30, T_i-5), T_i)$ is sampled. Features are causally generated up to $t_{\text{cutoff}}$, and the target is $RUL_{\text{true}} = T_i - t_{\text{cutoff}}$. Exactly one prediction per engine is evaluated per fold.
3. **Official Test Set Protection:** The official NASA test set (`test_FD001.txt` + `RUL_FD001.txt`) is preserved strictly as an untouched final evaluation benchmark.

### 3.4 NASA Asymmetric Prognostics Metric
In aviation, late predictions ($\hat{y} > y$) risk unexpected failure, while early predictions ($\hat{y} < y$) cause early inspections. The metric penalizes late estimates exponentially more severely:
$$d_i = \hat{y}_i - y_i$$
$$S = \sum_{i=1}^N s_i, \quad s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \end{cases}$$

---

## 4. Benchmark Results: Validation Endpoints vs. Official NASA Test Fleet

Model selection and target capping are guided strictly by **Simulated Validation Endpoint metrics**. Official test results are reported as a final post-selection benchmark.

| Exp ID | Model Architecture | Feature Set | Target Cap | Val Endpoint RMSE (cyc) | Val Endpoint NASA Score | All-Cycle CV RMSE (cyc) | Official Test RMSE (cyc) | Official Test MAE (cyc) | Official Test $R^2$ | Official Test NASA Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Median Baseline | Raw (14 sensors) | 125 | 57.23 | 1,077,635.8 | 44.70 | 49.82 | 39.08 | -0.437 | 166,570.5 |
| **EXP-02** | Linear Regression | Raw (14 sensors) | 125 | 35.28 | 475,724.9 | 21.31 | 21.91 | 17.61 | 0.722 | 1,318.9 |
| **EXP-03** | Linear Regression | Causal Temporal | 125 | 34.02 | 266,104.4 | 18.92 | 19.35 | 15.34 | 0.783 | 713.1 |
| **EXP-04** | Ridge Regression | Causal Temporal | 125 | 34.02 | 271,843.2 | 18.92 | 19.37 | 15.35 | 0.783 | 713.7 |
| **EXP-05** | Random Forest | Causal Temporal | 125 | 32.75 | 361,998.6 | 18.55 | 19.44 | 14.12 | 0.781 | 1,155.1 |
| **EXP-06** | **XGBoost (Champion)** | **Causal Temporal** | **125** | **32.38** | **253,898.6** | **17.61** | **17.86** | **13.28** | **0.815** | **678.7** |

---

## 5. Explainability & Uncertainty Quantification

### 5.1 TreeSHAP Feature Attribution
- **Dominant Statistical Drivers:** `s_11` ($Ps_{30}$: Static pressure at HPC outlet), `s_4` ($T_{50}$: Total temp at LPT outlet), and `s_12` ($\phi$: Ratio of fuel flow to $Ps_{30}$).
- **Local Explanations:** For critical engine inspections, SHAP waterfall plots quantify feature contributions pushing RUL predictions from the fleet base expectation down to terminal failure.

### 5.2 Split Conformal Prediction Intervals
- **Calibration Set:** 20 held-out training calibration engines (engine-disjoint; official test engines excluded).
- **Nominal Confidence Level:** 90% ($1 - \alpha = 0.90$).
- **Exact Finite-Sample Rank:** $k = \lceil(n+1)(1-\alpha)\rceil = 19 / 20$.
- **Conformal Margin $\pm q$:** $\pm 45.74$ cycles.
- **Empirical Test Fleet Coverage:** **82.0%** (honest reporting of the gap between uniform synthetic truncation and test-fleet inspection distribution).
- **Average Interval Width:** **85.6 cycles**.

---

## 6. Project Architecture

```text
AeroGuard/
|-- .gitignore
|-- requirements.txt
|-- Dockerfile
|-- .dockerignore
|-- README.md
|
|-- data/
|   |-- README.md
|   |-- raw/              # NASA C-MAPSS FD001 text files
|   `-- processed/
|
|-- notebooks/
|   |-- 01_eda_and_sensor_analysis.ipynb
|   `-- 02_model_benchmarking_and_evaluation.ipynb
|
|-- src/
|   |-- pipeline.py       # Unified end-to-end execution runner
|   |-- data/
|   |   `-- loader.py     # Deterministic parsing, schema validation, metadata
|   |-- features/
|   |   `-- engineer.py   # Causal rolling statistics & target capping
|   |-- models/
|   |   |-- train.py      # Pipelines, endpoint CV, all-cycle CV
|   |   |-- uncertainty.py# Exact finite-sample Split Conformal calibrator
|   |   |-- anomaly.py    # Isolation Forest anomaly detector
|   |   |-- run_benchmarks.py
|   |   `-- run_advanced.py
|   |-- evaluation/
|   |   `-- metrics.py    # MAE, RMSE, R2, NASA scoring metric
|   `-- explainability/
|       `-- shap_explainer.py # TreeSHAP explainability
|
|-- app/
|   `-- streamlit_app.py  # Defensive interactive operations dashboard
|
|-- models/               # Serialized pipeline artifacts (.joblib)
|-- results/
|   |-- metrics/          # CSV experiment logs
|   `-- figures/          # 15 high-resolution evaluation figures
|
|-- docs/
|   |-- methodology.md
|   |-- model_card.md
|   `-- limitations.md
|
`-- tests/
    |-- test_data_loader.py
    |-- test_feature_engineering.py
    |-- test_metrics.py
    |-- test_advanced.py
    `-- test_validation.py
```

---

## 7. How to Run & Verify

### Local Setup
```powershell
# 1. Clone repository and navigate
cd D:\projects\AeroGuard

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Run automated test suite (21 unit tests)
pytest tests -v

# 4. Run unified pipeline (reproduces models, calibration & figures)
python -m src.pipeline

# 5. Launch interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

### Docker Execution
```bash
# Build Docker image
docker build -t aeroguard:latest .

# Run container
docker run -p 8501:8501 aeroguard:latest
```
Open `http://localhost:8501` to access the fleet dashboard.

---

## 8. Limitations & Operational Scope
1. **Simulation Domain:** C-MAPSS models thermodynamic degradation via programmatic equations with Gaussian noise; real engines experience non-Gaussian bias, throttle transients, and external foreign object damage.
2. **Sea-Level Specificity (FD001):** The identified invariant sensors apply specifically to Sea Level single-condition operations. Multi-condition datasets (FD002, FD004) require flight-regime clustering.
3. **Decision-Support Status:** Risk tiers are heuristic decision-support thresholds to assist maintenance work-order scheduling, not FAA/EASA certified safety clearances.

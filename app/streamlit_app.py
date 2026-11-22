"""AeroGuard - Predictive Maintenance and Remaining Useful Life (RUL) Intelligence System.

Interactive Streamlit Application for Fleet Monitoring, RUL Inference,
Model Benchmarking, SHAP Explainability, and Conformal Uncertainty.
"""

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project modules are resolvable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.loader import load_fd001_raw, SENSOR_COLUMNS, SENSOR_METADATA
from src.evaluation.metrics import compute_regression_metrics
from src.explainability.shap_explainer import AeroGuardExplainer
from src.features.engineer import INFORMATIVE_SENSORS, compute_temporal_features

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroGuard | Turbofan RUL Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e3a8a; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
    .critical-badge { background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .warning-badge { background-color: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .healthy-badge { background-color: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Defensive Pipeline Verification
# -----------------------------------------------------------------------------
def check_prerequisites():
    raw_dir = project_root / "data" / "raw"
    models_dir = project_root / "models"
    metrics_dir = project_root / "results" / "metrics"

    missing = []
    for f in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
        if not (raw_dir / f).exists():
            missing.append(f"data/raw/{f}")
    for f in ["xgboost_best.joblib", "conformal_calibrator.joblib"]:
        if not (models_dir / f).exists():
            missing.append(f"models/{f}")
    if not (metrics_dir / "experiment_log.csv").exists():
        missing.append("results/metrics/experiment_log.csv")
    return missing

missing_files = check_prerequisites()
if missing_files:
    st.error("### ⚠️ Required Pipeline Artifacts Missing")
    st.markdown(
        "AeroGuard detected that raw data or trained artifacts are not yet generated:\n\n"
        + "\n".join([f"- `{m}`" for m in missing_files])
        + "\n\n**To reproduce the end-to-end pipeline (data verification, training, calibration, and metrics), run:**\n"
        "```powershell\npython -m src.pipeline\n```"
    )
    st.stop()


# -----------------------------------------------------------------------------
# Data & Model Resource Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    raw_dir = project_root / "data" / "raw"
    train_df, test_df, rul_series = load_fd001_raw(raw_dir)
    return train_df, test_df, rul_series


@st.cache_resource
def get_artifacts():
    models_dir = project_root / "models"
    xgb_artifact = joblib.load(models_dir / "xgboost_best.joblib")
    conformal = joblib.load(models_dir / "conformal_calibrator.joblib")
    exp_log = pd.read_csv(project_root / "results" / "metrics" / "experiment_log.csv")
    return xgb_artifact, conformal, exp_log


train_df, test_df, rul_series = get_dataset()
xgb_artifact, conformal_calibrator, experiment_log = get_artifacts()

pipeline = xgb_artifact["pipeline"]
feature_names = xgb_artifact["feature_names"]

# Compute test endpoint predictions
@st.cache_data
def get_test_predictions():
    test_proc = compute_temporal_features(test_df, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
    final_cycle_indices = test_proc.groupby("engine_id")["cycle"].idxmax()
    test_endpoints = test_proc.loc[final_cycle_indices].sort_values("engine_id").reset_index(drop=True)
    
    X_test = test_endpoints[feature_names]
    y_pred = np.maximum(0.0, pipeline.predict(X_test))
    y_true = rul_series.sort_index().to_numpy()
    
    lower, upper = conformal_calibrator.predict_interval(y_pred)
    
    df_results = pd.DataFrame({
        "engine_id": test_endpoints["engine_id"],
        "last_observed_cycle": test_endpoints["cycle"],
        "predicted_rul": np.round(y_pred, 1),
        "ci_lower_90": np.round(lower, 1),
        "ci_upper_90": np.round(upper, 1),
        "ground_truth_rul": y_true,
        "absolute_error": np.round(np.abs(y_pred - y_true), 1),
        "signed_error": np.round(y_pred - y_true, 1),
    })
    
    # Rule-based decision-support heuristic classification
    def assign_tier(rul):
        if rul <= 30:
            return "Critical (RUL \u2264 30)"
        elif rul <= 90:
            return "Warning (31-90)"
        return "Healthy (> 90)"
    
    df_results["risk_tier"] = df_results["predicted_rul"].apply(assign_tier)
    return test_endpoints, df_results


test_endpoints, test_results = get_test_predictions()

# -----------------------------------------------------------------------------
# Navigation & Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ✈️ AeroGuard Control")
view_option = st.sidebar.radio(
    "Navigation View:",
    [
        "1. Fleet Overview",
        "2. Individual Engine Health",
        "3. RUL Predictions & Risk Tiers",
        "4. Model Benchmarking",
        "5. SHAP Feature Attribution",
        "6. Conformal Uncertainty",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Dataset:** NASA C-MAPSS FD001  
    **Condition:** Sea Level (1 Operating Regime)  
    **Fault Mode:** HPC Degradation  
    **Fleet Size:** 100 Run-to-Failure / 100 Test  
    """
)
st.sidebar.info(
    "ℹ️ **Decision-Support Notice:** Predictions are computed using causal temporal feature engineering and calibrated conformal bounds. Risk tiers are heuristic decision-support thresholds."
)

# -----------------------------------------------------------------------------
# VIEW 1: FLEET OVERVIEW
# -----------------------------------------------------------------------------
if view_option == "1. Fleet Overview":
    st.markdown('<div class="main-title">AeroGuard: Fleet Maintenance Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Operational fleet health, lifetime distribution, and maintenance prioritization</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    crit_count = (test_results["risk_tier"].str.startswith("Critical")).sum()
    warn_count = (test_results["risk_tier"].str.startswith("Warning")).sum()
    healthy_count = (test_results["risk_tier"].str.startswith("Healthy")).sum()

    col1.metric("Active Test Fleet", f"{len(test_results)} Units")
    col2.metric("Critical Maintenance (\u226430 cyc)", f"{crit_count} Units", delta="Action Required", delta_color="inverse")
    col3.metric("Warning Inspection (31-90 cyc)", f"{warn_count} Units")
    col4.metric("Healthy Condition (>90 cyc)", f"{healthy_count} Units")

    st.markdown("### Fleet Operational Lifetime Profile")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(project_root / "results" / "figures" / "01_engine_lifespan_distribution.png"), caption="Fleet Lifecycle Distribution (Train vs. Test)")
    with col_b:
        st.image(str(project_root / "results" / "figures" / "05_population_normalized_degradation.png"), caption="Normalized Fleet Degradation Trajectory (% of Lifetime)")

# -----------------------------------------------------------------------------
# VIEW 2: INDIVIDUAL ENGINE HEALTH
# -----------------------------------------------------------------------------
elif view_option == "2. Individual Engine Health":
    st.markdown('<div class="main-title">Individual Engine Telemetry & Degradation Trajectory</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Historical sensor readings, temporal difference rates, and degradation monitoring</div>', unsafe_allow_html=True)

    dataset_choice = st.radio("Select Fleet Dataset:", ["Test Fleet (Truncated In-Flight)", "Training Fleet (Run-to-Failure)"], horizontal=True)
    target_df = test_df if dataset_choice.startswith("Test") else train_df

    available_engines = np.sort(target_df["engine_id"].unique())
    selected_engine = st.selectbox("Select Engine Identifier (Unit ID):", available_engines, index=0)

    engine_data = target_df[target_df["engine_id"] == selected_engine].sort_values("cycle")
    last_cycle = int(engine_data["cycle"].max())

    # Point summary metrics
    if dataset_choice.startswith("Test"):
        eng_res = test_results[test_results["engine_id"] == selected_engine].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Observed Cycle", f"{last_cycle} cyc")
        c2.metric("Predicted RUL", f"{eng_res['predicted_rul']:.1f} cyc")
        c3.metric("90% Conformal Interval", f"[{eng_res['ci_lower_90']:.0f}, {eng_res['ci_upper_90']:.0f}] cyc")
        c4.metric("Actual Ground-Truth RUL", f"{eng_res['ground_truth_rul']} cyc")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Observed Lifetime (Failure)", f"{last_cycle} cyc")
        c2.metric("Operating State", "Terminal Failure (RUL = 0)")

    st.markdown("---")
    st.markdown("### Interactive Sensor Telemetry")
    sensor_names_map = {
        col: f"{col} ({SENSOR_METADATA[col]['symbol']}: {SENSOR_METADATA[col]['description']} [{SENSOR_METADATA[col]['units']}])"
        for col in INFORMATIVE_SENSORS
        if col in SENSOR_METADATA
    }
    selected_sensors = st.multiselect(
        "Select Sensor Channels to Track:",
        options=list(sensor_names_map.keys()),
        default=["s_2", "s_4", "s_7", "s_11"],
        format_func=lambda x: sensor_names_map[x],
    )

    if selected_sensors:
        fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(11, 2.5 * len(selected_sensors)), dpi=180, sharex=True)
        if len(selected_sensors) == 1:
            axes = [axes]
        for ax, s in zip(axes, selected_sensors):
            ax.plot(engine_data["cycle"], engine_data[s], color="#1f77b4", linewidth=1.8, marker="o", markersize=2)
            # Add rolling mean trendline
            roll_mean = engine_data[s].rolling(10, min_periods=1).mean()
            ax.plot(engine_data["cycle"], roll_mean, color="#d62728", linestyle="--", linewidth=1.5, label="10-cycle Rolling Mean")
            ax.set_title(sensor_names_map[s], fontsize=10, fontweight="bold")
            ax.set_ylabel("Reading", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("Operating Cycle (t)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# -----------------------------------------------------------------------------
# VIEW 3: RUL PREDICTIONS & RISK TIERS
# -----------------------------------------------------------------------------
elif view_option == "3. RUL Predictions & Risk Tiers":
    st.markdown('<div class="main-title">Official Test Fleet RUL Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Endpoint predictions evaluated against official ground-truth targets (RUL_FD001.txt)</div>', unsafe_allow_html=True)

    test_metrics = compute_regression_metrics(test_results["ground_truth_rul"], test_results["predicted_rul"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Official Test RMSE", f"{test_metrics['rmse']:.2f} cyc")
    c2.metric("Official Test MAE", f"{test_metrics['mae']:.2f} cyc")
    c3.metric("Coefficient of Determination R\u00b2", f"{test_metrics['r2']:.3f}")
    c4.metric("NASA Asymmetric Score", f"{test_metrics['nasa_score']:.1f}")

    st.markdown("### Prediction vs. Ground-Truth Parity")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.image(str(project_root / "results" / "figures" / "07_actual_vs_predicted_rul.png"), caption="Predicted vs. Actual RUL Parity (Ridge vs. XGBoost)")
    with col_p2:
        st.image(str(project_root / "results" / "figures" / "09_error_by_degradation_stage.png"), caption="Prediction Error Breakdown Across Operating Stages")

    st.markdown("### Test Fleet Prediction Table")
    filter_tier = st.selectbox("Filter by Operational Risk Tier:", ["All Units", "Critical (\u2264 30)", "Warning (31-90)", "Healthy (> 90)"])
    filtered_results = test_results
    if "Critical" in filter_tier:
        filtered_results = test_results[test_results["risk_tier"].str.startswith("Critical")]
    elif "Warning" in filter_tier:
        filtered_results = test_results[test_results["risk_tier"].str.startswith("Warning")]
    elif "Healthy" in filter_tier:
        filtered_results = test_results[test_results["risk_tier"].str.startswith("Healthy")]

    st.dataframe(
        filtered_results[[
            "engine_id", "last_observed_cycle", "predicted_rul", 
            "ci_lower_90", "ci_upper_90", "ground_truth_rul", "absolute_error", "risk_tier"
        ]].rename(columns={
            "engine_id": "Engine ID",
            "last_observed_cycle": "Observed Cycles",
            "predicted_rul": "Predicted RUL",
            "ci_lower_90": "90% CI Lower",
            "ci_upper_90": "90% CI Upper",
            "ground_truth_rul": "Actual RUL",
            "absolute_error": "|Error| (cyc)",
            "risk_tier": "Assigned Risk Tier",
        }),
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# VIEW 4: MODEL BENCHMARKING
# -----------------------------------------------------------------------------
elif view_option == "4. Model Benchmarking":
    st.markdown('<div class="main-title">Prognostics Model Benchmarking & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Empirical comparison across models, features, and target capping strategies</div>', unsafe_allow_html=True)

    st.markdown("### Experiment Log (Engine-Level GroupKFold CV vs. Test Fleet)")
    st.dataframe(experiment_log, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(project_root / "results" / "figures" / "06_model_benchmark_comparison.png"), caption="Comparative Test Fleet Performance (RMSE & NASA Score)")
    with col2:
        st.image(str(project_root / "results" / "figures" / "10_target_cap_sensitivity.png"), caption="Target Capping Sensitivity Study (Performance vs. Ceiling)")

    st.markdown(
        r"""
        ### Key Mathematical Takeaways
        1. **Value of Temporal Features:** Incorporating causal rolling statistics ($\mu_{t,w}, \sigma_{t,w}$) and velocity ($\Delta_w x_t$) reduced the Linear Regression NASA Score from **1,318.9 to 713.1** ($46\%$ error reduction).
        2. **Nonlinear Gradient Boosting:** XGBoost achieved the best overall performance (**Test RMSE 17.86, Score 678.7**), effectively capturing the sharp exponential transition in the final 30% of engine lifecycle.
        3. **Target Capping Defense:** Uncapped regression achieves an unacceptable NASA score (**20,128.3**) due to severe penalization of noisy early-life predictions. Capping the target between **115 and 125 cycles** aligns with physical degradation onset and minimizes total asymmetric loss.
        """
    )

# -----------------------------------------------------------------------------
# VIEW 5: SHAP EXPLAINABILITY
# -----------------------------------------------------------------------------
elif view_option == "5. SHAP Feature Attribution":
    st.markdown('<div class="main-title">Explainable AI: Shapley Additive Explanations (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Global feature drivers and local game-theoretic prediction attribution</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Global Feature Importance", "Local Prediction Attribution (Single Engine)"])

    with tab1:
        st.image(str(project_root / "results" / "figures" / "11_shap_global_summary.png"), caption="Global TreeSHAP Summary Plot (Test Fleet)")
        st.markdown(
            r"""
            **Empirical Attribution Insights (Test Fleet TreeSHAP):**
            - **`s_11` features ($Ps_{30}$: Static pressure at HPC outlet):** Strongly influences RUL predictions as HPC pressure shift reflects degradation along the compression path.
            - **`s_4` features ($T_{50}$: Total temp at LPT outlet):** Higher exhaust gas temperatures statistically associate with reduced remaining life across tree splits.
            - **`s_12` features ($\phi$: Ratio of fuel flow to $Ps_{30}$):** Relative shift in fuel flow per unit static pressure serves as an empirical degradation indicator in the tree ensemble.
            *Note: SHAP values quantify feature contributions within the fitted model and reflect predictive association, not causal mechanics.*
            """
        )

    with tab2:
        st.markdown("#### Local Single-Engine Explanation")
        eng_select = st.selectbox("Select Test Engine for Inspection:", test_results["engine_id"].unique(), index=30)
        row_features = test_endpoints[test_endpoints["engine_id"] == eng_select].iloc[0]
        
        explainer = AeroGuardExplainer(pipeline, feature_names)
        local_exp = explainer.explain_engine(row_features)

        col_l1, col_l2 = st.columns([1, 2])
        with col_l1:
            st.metric("Base Fleet Value E[f(X)]", f"{local_exp['base_value']:.1f} cyc")
            st.metric("Engine Predicted RUL", f"{local_exp['prediction']:.1f} cyc")
            st.metric("Actual Ground-Truth RUL", f"{test_results[test_results['engine_id'] == eng_select]['ground_truth_rul'].values[0]} cyc")
        
        with col_l2:
            st.markdown("##### Top 8 Contributing Features for This Prediction")
            top_contr = local_exp["contributions"].head(8)
            st.dataframe(top_contr, use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW 6: CONFORMAL UNCERTAINTY
# -----------------------------------------------------------------------------
elif view_option == "6. Conformal Uncertainty":
    st.markdown('<div class="main-title">Conformal Prediction & Uncertainty Quantification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Distribution-free, finite-sample prediction intervals for maintenance decision-making</div>', unsafe_allow_html=True)

    cov_stats = conformal_calibrator.evaluate_coverage(test_results["ground_truth_rul"], test_results["predicted_rul"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Nominal Confidence Level", f"{cov_stats['nominal_coverage']*100:.0f}%")
    c2.metric("Empirical Test Fleet Coverage", f"{cov_stats['empirical_coverage']*100:.1f}%")
    c3.metric("Conformal Quantile Margin \u00b1q", f"\u00b1{conformal_calibrator.quantile_:.1f} cyc")

    st.image(str(project_root / "results" / "figures" / "14_conformal_prediction_intervals.png"), caption="90% Conformal Prediction Intervals Across All 100 Test Engines")
    st.image(str(project_root / "results" / "figures" / "15_anomaly_score_trajectory.png"), caption="Complementary Isolation Forest Anomaly Progression Over Engine Lifetime")

    st.markdown(
        r"""
        ### Rigorous Mathematical Grounding
        Unlike heuristic Gaussian assumptions ($\hat{y} \pm 1.96\sigma$), **Split Conformal Prediction** guarantees valid coverage under minimal assumptions:
        - **Assumption:** Exchangeability of calibration and test error distributions (both drawn from C-MAPSS FD001 operating regime).
        - **Guarantee:** $\mathbb{P}(y \in C(X)) \ge 1 - \alpha$.
        - **Decision-Making Benefit:** Maintenance managers receive conservative lower-bound bounds ($\text{RUL}_{\text{lower}}$) for scheduling component replacements before the 90% confidence threshold is breached.
        """
    )

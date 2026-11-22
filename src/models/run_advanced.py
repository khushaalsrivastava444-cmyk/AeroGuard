"""Advanced analysis runner: SHAP explainability, conformal uncertainty, and anomaly detection."""

import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.loader import load_fd001_raw
from src.explainability.shap_explainer import AeroGuardExplainer
from src.features.engineer import (
    INFORMATIVE_SENSORS,
    compute_temporal_features,
    prepare_feature_matrix,
)
from src.models.anomaly import EngineAnomalyDetector
from src.models.uncertainty import SplitConformalPredictor

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"


def run_advanced_experiments():
    print("=" * 70)
    print("AeroGuard: Milestone 7, 8 — Explainability, Uncertainty & Anomaly")
    print("=" * 70)

    train_df, test_df, rul_series = load_fd001_raw(project_root / "data" / "raw")
    figures_dir = project_root / "results" / "figures"
    models_dir = project_root / "models"
    metrics_dir = project_root / "results" / "metrics"

    # Load trained XGBoost artifact
    xgb_artifact = joblib.load(models_dir / "xgboost_best.joblib")
    pipeline = xgb_artifact["pipeline"]
    feature_names = xgb_artifact["feature_names"]

    # Prepare test data at final cycle
    test_proc = compute_temporal_features(test_df, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
    final_cycle_indices = test_proc.groupby("engine_id")["cycle"].idxmax()
    test_last_cycles = test_proc.loc[final_cycle_indices].sort_values("engine_id").reset_index(drop=True)
    X_test_last = test_last_cycles[feature_names]
    y_test_true = rul_series.sort_index().to_numpy()
    y_test_pred = np.maximum(0.0, pipeline.predict(X_test_last))

    # -------------------------------------------------------------
    # 1. SHAP Explainability
    # -------------------------------------------------------------
    print("\n--- Computing TreeSHAP Values on Test Fleet ---")
    explainer = AeroGuardExplainer(pipeline, feature_names)
    explanation = explainer.explain(X_test_last)

    # Figure 11: SHAP Global Summary (Beeswarm)
    fig = plt.figure(figsize=(11, 7), dpi=200)
    shap.summary_plot(explanation, X_test_last, show=False, max_display=15)
    plt.title("AeroGuard: SHAP Global Feature Attributions (Test Fleet)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "11_shap_global_summary.png")
    plt.close()

    # Find a critical engine (lowest true RUL) and a healthy engine (highest true RUL)
    critical_idx = int(np.argmin(y_test_true))  # e.g. RUL = 7 or 8
    healthy_idx = int(np.argmax(y_test_true))   # e.g. RUL = 145

    crit_engine_id = test_last_cycles.iloc[critical_idx]["engine_id"]
    healthy_engine_id = test_last_cycles.iloc[healthy_idx]["engine_id"]

    # Figure 12: Waterfall plot for Critical Engine
    fig = plt.figure(figsize=(10, 5), dpi=200)
    shap.plots.waterfall(explanation[critical_idx], max_display=10, show=False)
    plt.title(f"SHAP Waterfall: Engine {crit_engine_id} in Critical State (Actual RUL: {y_test_true[critical_idx]}, Pred: {y_test_pred[critical_idx]:.1f})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "12_shap_waterfall_critical_engine.png")
    plt.close()

    # Figure 13: Waterfall plot for Healthy Engine
    fig = plt.figure(figsize=(10, 5), dpi=200)
    shap.plots.waterfall(explanation[healthy_idx], max_display=10, show=False)
    plt.title(f"SHAP Waterfall: Engine {healthy_engine_id} in Healthy State (Actual RUL: {y_test_true[healthy_idx]}, Pred: {y_test_pred[healthy_idx]:.1f})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_shap_waterfall_healthy_engine.png")
    plt.close()

    # -------------------------------------------------------------
    # 2. Conformal Uncertainty Estimation
    # -------------------------------------------------------------
    print("\n--- Calibrating Split Conformal Predictor (90% Nominal Coverage) ---")
    # Hold out 20 training engines strictly for calibration (engine-disjoint split)
    rng = np.random.RandomState(42)
    all_train_engines = np.sort(train_df["engine_id"].unique())
    cal_engine_ids = rng.choice(all_train_engines, size=20, replace=False)
    proper_train_ids = np.setdiff1d(all_train_engines, cal_engine_ids)

    df_proper = train_df[train_df["engine_id"].isin(proper_train_ids)].copy()
    df_cal = train_df[train_df["engine_id"].isin(cal_engine_ids)].copy()

    # Fit pipeline strictly on proper training fleet (80 engines)
    X_proper, y_proper, _ = prepare_feature_matrix(df_proper, include_temporal=True, rul_cap=125, is_train=True)
    cal_pipe = pipeline
    cal_pipe.fit(X_proper, y_proper)

    # Simulate operational truncated inspection endpoints across calibration engines
    # (mirroring the test evaluation setup where each engine is inspected at an unknown cutoff)
    cal_endpoint_rows = []
    cal_targets = []
    for eng_id, group in df_cal.groupby("engine_id"):
        t_max = int(group["cycle"].max())
        cutoff = rng.randint(min(30, t_max - 5), t_max)
        truncated_group = group[group["cycle"] <= cutoff].copy()
        features = compute_temporal_features(truncated_group, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
        cal_endpoint_rows.append(features.iloc[[-1]])
        cal_targets.append(t_max - cutoff)

    cal_endpoint_df = pd.concat(cal_endpoint_rows, ignore_index=True)
    X_cal_endpoint = cal_endpoint_df[feature_names]
    y_cal_true = np.array(cal_targets)
    y_cal_pred = np.maximum(0.0, cal_pipe.predict(X_cal_endpoint))

    conformal = SplitConformalPredictor(alpha=0.10)  # 90% target coverage
    conformal.fit(y_cal_true, y_cal_pred)

    cov_results = conformal.evaluate_coverage(y_test_true, y_test_pred)
    print(f"Calibration Engines: {len(y_cal_true)} (1 simulated cutoff per engine)")
    print(f"Finite-Sample Quantile Index k: {int(np.ceil((len(y_cal_true) + 1) * 0.90))} / {len(y_cal_true)}")
    print(f"Conformal Calibration Margin q: \u00b1{conformal.quantile_:.2f} cycles")
    print(f"Empirical Test Coverage (Nominal 90%): {cov_results['empirical_coverage']*100:.1f}%")
    print(f"Mean Interval Width: {cov_results['mean_interval_width']:.1f} cycles")

    joblib.dump(conformal, models_dir / "conformal_calibrator.joblib")

    # Figure 14: Conformal Prediction Intervals on Test Set
    lower_bounds, upper_bounds = conformal.predict_interval(y_test_pred)
    sort_idx = np.argsort(y_test_true)

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)
    x_axis = np.arange(len(y_test_true))
    ax.fill_between(x_axis, lower_bounds[sort_idx], upper_bounds[sort_idx], color="#1f77b4", alpha=0.25, label=f"90% Conformal Interval (\u00b1{conformal.quantile_:.1f} cyc)")
    ax.plot(x_axis, y_test_true[sort_idx], "k-", linewidth=2.0, label="Actual Ground-Truth RUL")
    ax.scatter(x_axis, y_test_pred[sort_idx], color="#d62728", s=18, label="Predicted RUL \u0177", zorder=3)
    ax.set_title(f"AeroGuard: 90% Conformal Prediction Intervals Across Test Fleet (Coverage: {cov_results['empirical_coverage']*100:.1f}%)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Test Engines (Ordered by True RUL)", fontsize=10)
    ax.set_ylabel("Remaining Useful Life (cycles)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "14_conformal_prediction_intervals.png")
    plt.close()

    # -------------------------------------------------------------
    # 3. Anomaly Detection (Isolation Forest)
    # -------------------------------------------------------------
    print("\n--- Fitting Isolation Forest Anomaly Detector ---")
    detector = EngineAnomalyDetector(n_estimators=100, contamination=0.10, random_state=42)
    X_train_full, _, feat_cols_det = prepare_feature_matrix(train_df, include_temporal=True, rul_cap=125, is_train=False)
    detector.fit(X_train_full)
    joblib.dump(detector, models_dir / "anomaly_detector.joblib")

    # Track anomaly score along sample engine trajectory (Engine 39 - shortest life)
    eng_sample = train_df[train_df["engine_id"] == 39].copy()
    eng_sample_feat = compute_temporal_features(eng_sample, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
    anomaly_scores = detector.score_samples(eng_sample_feat[feat_cols_det])
    eng_sample["anomaly_score"] = anomaly_scores

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), dpi=200, sharex=True)
    ax1.plot(eng_sample["cycle"], eng_sample["s_4"], color="#1f77b4", linewidth=1.8, label="s_4 (T50: Total temp at LPT outlet [°R])")
    ax1.set_title("Engine 39 (Failure at Cycle 128): Physical Sensor Degradation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Temp [R]", fontsize=10)
    ax1.legend(loc="upper left")

    ax2.plot(eng_sample["cycle"], eng_sample["anomaly_score"], color="#d62728", linewidth=2.0, label="Isolation Anomaly Score s(x)")
    ax2.axhline(0.5, color="gray", linestyle="--", label="Nominal Operational Baseline (0.5)")
    ax2.set_title("Isolation Forest Anomaly Progression Over Operational Lifetime", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Operating Cycle", fontsize=10)
    ax2.set_ylabel("Anomaly Score [0, 1]", fontsize=10)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(figures_dir / "15_anomaly_score_trajectory.png")
    plt.close()

    print("Advanced Experiments Completed Successfully!")


if __name__ == "__main__":
    run_advanced_experiments()

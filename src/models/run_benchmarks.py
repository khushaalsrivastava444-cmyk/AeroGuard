"""Comprehensive model benchmarking, target cap ablation, and residual evaluation."""

import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.loader import load_fd001_raw
from src.features.engineer import prepare_feature_matrix
from src.models.train import (
    build_pipeline,
    evaluate_on_test_fleet,
    run_engine_group_cv,
    train_and_save_model,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"


def run_all_benchmarks():
    print("=" * 70)
    print("AeroGuard: Milestone 4, 5, 6 — Model Benchmarking & Validation")
    print("=" * 70)

    train_df, test_df, rul_series = load_fd001_raw(project_root / "data" / "raw")
    engine_ids = train_df["engine_id"]

    results_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    models_dir = project_root / "models"
    for d in (results_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    experiments = []
    trained_artifacts = {}

    # Define model configurations to benchmark
    benchmark_configs = [
        {"id": "EXP-01", "name": "median", "features": "raw", "temporal": False, "cap": 125, "kwargs": {}},
        {"id": "EXP-02", "name": "linear", "features": "raw", "temporal": False, "cap": 125, "kwargs": {}},
        {"id": "EXP-03", "name": "linear", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {}},
        {"id": "EXP-04", "name": "ridge", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"alpha": 10.0}},
        {"id": "EXP-05", "name": "rf", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"n_estimators": 100, "max_depth": 12}},
        {"id": "EXP-06", "name": "xgb", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.05}},
    ]

    print("\n--- Running Core Model Benchmarks (5-Fold Engine GroupKFold) ---")
    predictions_dict = {}

    for cfg in benchmark_configs:
        exp_id = cfg["id"]
        model_name = cfg["name"]
        is_temp = cfg["temporal"]
        cap = cfg["cap"]
        kwargs = cfg["kwargs"]

        print(f"[{exp_id}] Model: {model_name.upper()} | Features: {cfg['features']} | Target Cap: {cap}...")

        # 1. Feature preparation
        X_train, y_train, feat_cols = prepare_feature_matrix(
            train_df, include_temporal=is_temp, rul_cap=cap, is_train=True
        )

        # 2. Engine-level Group Cross-Validation
        pipe = build_pipeline(model_name, **kwargs)
        cv_res = run_engine_group_cv(pipe, X_train, y_train, engine_ids=engine_ids, n_splits=5)

        # 3. Fit on full training fleet and evaluate on official test set
        pipe.fit(X_train, y_train)
        test_metrics, y_true, y_pred = evaluate_on_test_fleet(
            pipe, test_df, rul_series, feature_names=feat_cols, include_temporal=is_temp
        )

        predictions_dict[exp_id] = {"y_true": y_true, "y_pred": y_pred, "name": model_name}

        # 4. Save model artifact
        model_filename = f"{model_name}_{cfg['features']}.joblib"
        save_path = models_dir / model_filename
        joblib.dump(
            {
                "model_name": model_name,
                "pipeline": pipe,
                "feature_names": feat_cols,
                "rul_cap": cap,
                "include_temporal": is_temp,
                "test_metrics": test_metrics,
            },
            save_path,
        )
        trained_artifacts[exp_id] = save_path

        exp_record = {
            "Experiment_ID": exp_id,
            "Model": model_name.upper(),
            "Features": cfg["features"],
            "Target_Cap": cap,
            "CV_MAE": round(cv_res["cv_mae_mean"], 2),
            "CV_RMSE": round(cv_res["cv_rmse_mean"], 2),
            "CV_R2": round(cv_res["cv_r2_mean"], 3),
            "Test_MAE": round(test_metrics["mae"], 2),
            "Test_RMSE": round(test_metrics["rmse"], 2),
            "Test_R2": round(test_metrics["r2"], 3),
            "Test_NASA_Score": round(test_metrics["nasa_score"], 1),
        }
        experiments.append(exp_record)
        print(f"    -> Test RMSE: {test_metrics['rmse']:.2f} | Test MAE: {test_metrics['mae']:.2f} | NASA Score: {test_metrics['nasa_score']:.1f}")

    # -------------------------------------------------------------
    # Target Capping Sensitivity Analysis (XGBoost)
    # -------------------------------------------------------------
    print("\n--- Running Target Cap Sensitivity Analysis (XGBoost) ---")
    cap_candidates = [None, 100, 115, 125, 130, 140, 150]
    cap_results = []

    for cap in cap_candidates:
        cap_label = "Uncapped" if cap is None else str(cap)
        print(f"[CAP STUDY] XGBoost with Target Cap = {cap_label}...")

        X_train, y_train, feat_cols = prepare_feature_matrix(
            train_df, include_temporal=True, rul_cap=cap, is_train=True
        )

        pipe = build_pipeline("xgb", n_estimators=120, max_depth=5, learning_rate=0.05)
        pipe.fit(X_train, y_train)

        test_metrics, _, _ = evaluate_on_test_fleet(
            pipe, test_df, rul_series, feature_names=feat_cols, include_temporal=True
        )

        cap_results.append({
            "RUL_Cap": cap if cap is not None else 200,
            "Cap_Label": cap_label,
            "Test_MAE": test_metrics["mae"],
            "Test_RMSE": test_metrics["rmse"],
            "Test_R2": test_metrics["r2"],
            "Test_NASA_Score": test_metrics["nasa_score"],
        })
        print(f"    -> Test RMSE: {test_metrics['rmse']:.2f} | NASA Score: {test_metrics['nasa_score']:.1f}")

    # Save best XGBoost artifact
    best_xgb_path = models_dir / "xgboost_best.joblib"
    X_train_best, y_train_best, best_feats = prepare_feature_matrix(
        train_df, include_temporal=True, rul_cap=125, is_train=True
    )
    best_pipe = build_pipeline("xgb", n_estimators=120, max_depth=5, learning_rate=0.05)
    best_pipe.fit(X_train_best, y_train_best)
    best_metrics, _, _ = evaluate_on_test_fleet(
        best_pipe, test_df, rul_series, feature_names=best_feats, include_temporal=True
    )
    joblib.dump(
        {
            "model_name": "xgboost",
            "pipeline": best_pipe,
            "feature_names": best_feats,
            "rul_cap": 125,
            "include_temporal": True,
            "test_metrics": best_metrics,
        },
        best_xgb_path,
    )

    # -------------------------------------------------------------
    # Persist Metric Logs
    # -------------------------------------------------------------
    df_exp = pd.DataFrame(experiments)
    df_exp.to_csv(results_dir / "experiment_log.csv", index=False)
    print(f"\nExperiment log written to {results_dir / 'experiment_log.csv'}")

    df_cap = pd.DataFrame(cap_results)
    df_cap.to_csv(results_dir / "target_cap_sensitivity.csv", index=False)

    # -------------------------------------------------------------
    # Visualization Figures
    # -------------------------------------------------------------
    print("\n--- Generating Evaluation Figures in results/figures/ ---")

    # Figure 6: Benchmark comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    labels = [f"{r['Model']}\n({r['Features']})" for r in experiments]

    bars1 = ax1.bar(labels, [r["Test_RMSE"] for r in experiments], color="#1f77b4", width=0.55)
    ax1.set_title("Official Test Set: Root Mean Squared Error (RMSE)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (cycles)", fontsize=10)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", fontsize=9)

    bars2 = ax2.bar(labels, [r["Test_NASA_Score"] for r in experiments], color="#d62728", width=0.55)
    ax2.set_title("Official Test Set: NASA Prognostics Score (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total NASA Score", fontsize=10)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 50, f"{int(b.get_height())}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "06_model_benchmark_comparison.png")
    plt.close()

    # Figure 7: Actual vs Predicted RUL (XGBoost vs Ridge)
    xgb_pred = predictions_dict["EXP-06"]["y_pred"]
    ridge_pred = predictions_dict["EXP-04"]["y_pred"]
    y_test_true = predictions_dict["EXP-06"]["y_true"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200, sharey=True)

    ax1.scatter(y_test_true, ridge_pred, alpha=0.75, color="#2ca02c", edgecolors="k", linewidth=0.5)
    ax1.plot([0, 160], [0, 160], "r--", linewidth=1.5, label="Ideal Parity y = x")
    ax1.set_title(f"Ridge Regression (Test RMSE: {experiments[3]['Test_RMSE']:.1f}, Score: {experiments[3]['Test_NASA_Score']:.0f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Actual Ground-Truth RUL", fontsize=10)
    ax1.set_ylabel("Predicted RUL", fontsize=10)
    ax1.legend()

    ax2.scatter(y_test_true, xgb_pred, alpha=0.8, color="#1f77b4", edgecolors="k", linewidth=0.5)
    ax2.plot([0, 160], [0, 160], "r--", linewidth=1.5, label="Ideal Parity y = x")
    ax2.set_title(f"XGBoost Regressor (Test RMSE: {experiments[5]['Test_RMSE']:.1f}, Score: {experiments[5]['Test_NASA_Score']:.0f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Actual Ground-Truth RUL", fontsize=10)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "07_actual_vs_predicted_rul.png")
    plt.close()

    # Figure 8: Residual Error Distribution
    residuals_xgb = xgb_pred - y_test_true
    residuals_ridge = ridge_pred - y_test_true

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    sns.histplot(residuals_xgb, bins=20, kde=True, color="#1f77b4", label=f"XGBoost (\u03bc={residuals_xgb.mean():.1f}, \u03c3={residuals_xgb.std():.1f})", alpha=0.5, ax=ax)
    sns.histplot(residuals_ridge, bins=20, kde=True, color="#2ca02c", label=f"Ridge (\u03bc={residuals_ridge.mean():.1f}, \u03c3={residuals_ridge.std():.1f})", alpha=0.4, ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Error Parity")
    ax.set_title("Test Set Prediction Residuals Distribution: e = y_hat - y", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual Error (Predicted - Actual RUL)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(figures_dir / "08_residual_error_distribution.png")
    plt.close()

    # Figure 9: Error by Degradation Stage (Critical <= 30, Warning 31-90, Healthy > 90)
    stages = []
    for y_val in y_test_true:
        if y_val <= 30:
            stages.append("Critical (RUL \u2264 30)")
        elif y_val <= 90:
            stages.append("Warning (31 \u2264 RUL \u2264 90)")
        else:
            stages.append("Healthy (RUL > 90)")

    df_errors = pd.DataFrame({
        "Stage": stages,
        "Actual_RUL": y_test_true,
        "Absolute_Error": np.abs(residuals_xgb),
        "Signed_Error": residuals_xgb,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    sns.boxplot(data=df_errors, x="Stage", y="Absolute_Error", palette="Blues", ax=ax1, width=0.45)
    ax1.set_title("XGBoost: Absolute Error by Operating Stage", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Absolute Error |y - y_hat|", fontsize=10)

    sns.boxplot(data=df_errors, x="Stage", y="Signed_Error", palette="vlag", ax=ax2, width=0.45)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax2.set_title("XGBoost: Signed Residual by Operating Stage", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Signed Error (y_hat - y)", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "09_error_by_degradation_stage.png")
    plt.close()

    # Figure 10: Target Capping Sensitivity Curve
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=200)
    caps_plot = [r["RUL_Cap"] for r in cap_results if r["RUL_Cap"] <= 150]
    rmse_plot = [r["Test_RMSE"] for r in cap_results if r["RUL_Cap"] <= 150]
    nasa_plot = [r["Test_NASA_Score"] for r in cap_results if r["RUL_Cap"] <= 150]

    color = "#1f77b4"
    ax1.set_xlabel("RUL Target Capping Threshold (cycles)", fontsize=10)
    ax1.set_ylabel("Test Set RMSE (cycles)", color=color, fontsize=10)
    line1 = ax1.plot(caps_plot, rmse_plot, marker="o", color=color, linewidth=2, label="Test RMSE")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("NASA Prognostics Score", color=color, fontsize=10)
    line2 = ax2.plot(caps_plot, nasa_plot, marker="s", color=color, linewidth=2, linestyle="--", label="NASA Score")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Target Cap Sensitivity: Performance vs. RUL Ceiling Threshold", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "10_target_cap_sensitivity.png")
    plt.close()

    print("\nBenchmark Execution Complete! Summary Table:")
    print(df_exp.to_string(index=False))


if __name__ == "__main__":
    run_all_benchmarks()

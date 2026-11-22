"""Comprehensive model benchmarking, target cap ablation, and residual evaluation.

Model selection and target capping are performed STRICTLY via engine-level
endpoint cross-validation. The official NASA test set is preserved purely for
final, unbiased benchmark reporting.
"""

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

from src.data.loader import SENSOR_METADATA, load_fd001_raw
from src.features.engineer import prepare_feature_matrix
from src.models.train import (
    build_pipeline,
    evaluate_on_test_fleet,
    run_all_cycle_cv,
    run_endpoint_group_cv,
    train_and_save_model,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"


def run_all_benchmarks():
    print("=" * 75)
    print("  AeroGuard: Model Benchmarking & Validation Protocol")
    print("  Selection based strictly on held-out Engine Endpoint Validation")
    print("=" * 75)

    train_df, test_df, rul_series = load_fd001_raw(project_root / "data" / "raw")
    engine_ids = train_df["engine_id"]

    results_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    models_dir = project_root / "models"
    for d in (results_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    experiments = []
    predictions_dict = {}

    # Benchmark configurations
    benchmark_configs = [
        {"id": "EXP-01", "name": "median", "features": "raw", "temporal": False, "cap": 125, "kwargs": {}},
        {"id": "EXP-02", "name": "linear", "features": "raw", "temporal": False, "cap": 125, "kwargs": {}},
        {"id": "EXP-03", "name": "linear", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {}},
        {"id": "EXP-04", "name": "ridge", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"alpha": 10.0}},
        {"id": "EXP-05", "name": "rf", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"n_estimators": 100, "max_depth": 12}},
        {"id": "EXP-06", "name": "xgb", "features": "temporal", "temporal": True, "cap": 125, "kwargs": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.05}},
    ]

    print("\n[PHASE 1] Running Engine-Level Cross-Validation (Endpoint & All-Cycle)...")

    for cfg in benchmark_configs:
        exp_id = cfg["id"]
        model_name = cfg["name"]
        is_temp = cfg["temporal"]
        cap = cfg["cap"]
        kwargs = cfg["kwargs"]

        print(f"  [{exp_id}] Model: {model_name.upper():<7} | Features: {cfg['features']:<8} | Cap: {cap}...")

        pipe = build_pipeline(model_name, **kwargs)

        # 1. Primary Selection Metric: Engine Endpoint GroupKFold (simulating operational test task)
        endpoint_cv = run_endpoint_group_cv(
            pipeline=pipe,
            train_df=train_df,
            include_temporal=is_temp,
            rul_cap=cap,
            n_splits=5,
            seed=42,
            min_observed_cycles=30,
        )

        # 2. Secondary Diagnostic Metric: All-Cycle GroupKFold
        X_train, y_train, feat_cols = prepare_feature_matrix(
            train_df, include_temporal=is_temp, rul_cap=cap, is_train=True
        )
        all_cycle_cv = run_all_cycle_cv(pipe, X_train, y_train, engine_ids=engine_ids, n_splits=5)

        # 3. Fit on full training fleet for final model artifact
        pipe.fit(X_train, y_train)

        # 4. Final Untouched Official Test Evaluation
        test_metrics, y_true, y_pred = evaluate_on_test_fleet(
            pipe, test_df, rul_series, feature_names=feat_cols, include_temporal=is_temp
        )
        predictions_dict[exp_id] = {"y_true": y_true, "y_pred": y_pred, "name": model_name}

        # 5. Persist model artifact
        model_filename = f"{model_name}_{cfg['features']}.joblib"
        joblib.dump(
            {
                "model_name": model_name,
                "pipeline": pipe,
                "feature_names": feat_cols,
                "rul_cap": cap,
                "include_temporal": is_temp,
                "val_endpoint_metrics": endpoint_cv,
                "all_cycle_metrics": all_cycle_cv,
                "official_test_metrics": test_metrics,
            },
            models_dir / model_filename,
        )

        exp_record = {
            "Experiment_ID": exp_id,
            "Model": model_name.upper(),
            "Features": cfg["features"],
            "Target_Cap": cap,
            "Val_Endpoint_RMSE": round(endpoint_cv["val_endpoint_rmse"], 2),
            "Val_Endpoint_MAE": round(endpoint_cv["val_endpoint_mae"], 2),
            "Val_Endpoint_NASA": round(endpoint_cv["val_endpoint_nasa"], 1),
            "All_Cycle_RMSE": round(all_cycle_cv["all_cycle_rmse"], 2),
            "Official_Test_RMSE": round(test_metrics["rmse"], 2),
            "Official_Test_MAE": round(test_metrics["mae"], 2),
            "Official_Test_R2": round(test_metrics["r2"], 3),
            "Official_Test_NASA": round(test_metrics["nasa_score"], 1),
        }
        experiments.append(exp_record)
        print(f"       -> Val Endpoint RMSE: {endpoint_cv['val_endpoint_rmse']:.2f} | Val NASA: {endpoint_cv['val_endpoint_nasa']:.1f}")
        print(f"       -> Official Test RMSE: {test_metrics['rmse']:.2f} | Test NASA: {test_metrics['nasa_score']:.1f}")

    # -------------------------------------------------------------
    # [PHASE 2] Target Capping Sensitivity Analysis (Purely via Endpoint CV)
    # -------------------------------------------------------------
    print("\n[PHASE 2] Target Capping Sensitivity Analysis (Evaluated on Validation Endpoints)...")
    cap_candidates = [None, 100, 115, 125, 130, 140, 150]
    cap_results = []

    for cap in cap_candidates:
        cap_label = "Uncapped" if cap is None else str(cap)
        print(f"  [Cap Study] XGBoost with Target Cap = {cap_label:<8} (Running 5-Fold Endpoint CV)...")

        pipe = build_pipeline("xgb", n_estimators=120, max_depth=5, learning_rate=0.05)
        endpoint_cv = run_endpoint_group_cv(
            pipeline=pipe,
            train_df=train_df,
            include_temporal=True,
            rul_cap=cap,
            n_splits=5,
            seed=42,
            min_observed_cycles=30,
        )

        # Evaluate on test set for reporting only
        X_train, y_train, feat_cols = prepare_feature_matrix(
            train_df, include_temporal=True, rul_cap=cap, is_train=True
        )
        pipe.fit(X_train, y_train)
        test_metrics, _, _ = evaluate_on_test_fleet(
            pipe, test_df, rul_series, feature_names=feat_cols, include_temporal=True
        )

        cap_results.append({
            "RUL_Cap": cap if cap is not None else 200,
            "Cap_Label": cap_label,
            "Val_Endpoint_RMSE": endpoint_cv["val_endpoint_rmse"],
            "Val_Endpoint_MAE": endpoint_cv["val_endpoint_mae"],
            "Val_Endpoint_NASA": endpoint_cv["val_endpoint_nasa"],
            "Official_Test_RMSE": test_metrics["rmse"],
            "Official_Test_NASA": test_metrics["nasa_score"],
        })
        print(f"       -> Val Endpoint RMSE: {endpoint_cv['val_endpoint_rmse']:.2f} | Val NASA: {endpoint_cv['val_endpoint_nasa']:.1f}")

    # Determine best cap based strictly on lowest Val_Endpoint_NASA
    df_cap = pd.DataFrame(cap_results)
    best_cap_row = df_cap.loc[df_cap["Val_Endpoint_NASA"].idxmin()]
    best_selected_cap = int(best_cap_row["RUL_Cap"]) if best_cap_row["RUL_Cap"] != 200 else None
    print(f"\n  [Selection Result] Validation Endpoint minimum NASA score selects Target Cap = {best_cap_row['Cap_Label']}")

    # Save champion XGBoost artifact with validation-selected cap
    X_train_best, y_train_best, best_feats = prepare_feature_matrix(
        train_df, include_temporal=True, rul_cap=best_selected_cap, is_train=True
    )
    best_pipe = build_pipeline("xgb", n_estimators=120, max_depth=5, learning_rate=0.05)
    best_pipe.fit(X_train_best, y_train_best)
    best_test_metrics, _, _ = evaluate_on_test_fleet(
        best_pipe, test_df, rul_series, feature_names=best_feats, include_temporal=True
    )
    joblib.dump(
        {
            "model_name": "xgboost",
            "pipeline": best_pipe,
            "feature_names": best_feats,
            "rul_cap": best_selected_cap,
            "include_temporal": True,
            "official_test_metrics": best_test_metrics,
        },
        models_dir / "xgboost_best.joblib",
    )

    # Persist CSV logs
    df_exp = pd.DataFrame(experiments)
    df_exp.to_csv(results_dir / "experiment_log.csv", index=False)
    df_cap.to_csv(results_dir / "target_cap_sensitivity.csv", index=False)
    print(f"\nExperiment log written to {results_dir / 'experiment_log.csv'}")

    # -------------------------------------------------------------
    # Visualizations with Corrected Labels
    # -------------------------------------------------------------
    print("\n[PHASE 3] Generating Evaluation Figures in results/figures/...")

    # Figure 6: Benchmark comparison bar chart (Validation Endpoint vs Official Test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    labels = [f"{r['Model']}\n({r['Features']})" for r in experiments]
    x_arr = np.arange(len(labels))
    width = 0.35

    ax1.bar(x_arr - width / 2, [r["Val_Endpoint_RMSE"] for r in experiments], width, label="Val Endpoint RMSE", color="#2ca02c")
    ax1.bar(x_arr + width / 2, [r["Official_Test_RMSE"] for r in experiments], width, label="Official Test RMSE", color="#1f77b4")
    ax1.set_xticks(x_arr)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Root Mean Squared Error (RMSE): Validation vs. Test", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (cycles)", fontsize=10)
    ax1.legend()

    ax2.bar(x_arr - width / 2, [r["Val_Endpoint_NASA"] for r in experiments], width, label="Val Endpoint NASA", color="#ff7f0e")
    ax2.bar(x_arr + width / 2, [r["Official_Test_NASA"] for r in experiments], width, label="Official Test NASA", color="#d62728")
    ax2.set_xticks(x_arr)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("NASA Prognostics Score: Validation vs. Test (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Score", fontsize=10)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "06_model_benchmark_comparison.png")
    plt.close()

    # Figure 7: Actual vs Predicted RUL (Ridge vs XGBoost)
    xgb_pred = predictions_dict["EXP-06"]["y_pred"]
    ridge_pred = predictions_dict["EXP-04"]["y_pred"]
    y_test_true = predictions_dict["EXP-06"]["y_true"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200, sharey=True)
    ax1.scatter(y_test_true, ridge_pred, alpha=0.75, color="#2ca02c", edgecolors="k", linewidth=0.5)
    ax1.plot([0, 160], [0, 160], "r--", linewidth=1.5, label="Ideal Parity y = x")
    ax1.set_title(f"Ridge Regression (Test RMSE: {experiments[3]['Official_Test_RMSE']:.1f}, NASA: {experiments[3]['Official_Test_NASA']:.0f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Actual Ground-Truth RUL", fontsize=10)
    ax1.set_ylabel("Predicted RUL", fontsize=10)
    ax1.legend()

    ax2.scatter(y_test_true, xgb_pred, alpha=0.8, color="#1f77b4", edgecolors="k", linewidth=0.5)
    ax2.plot([0, 160], [0, 160], "r--", linewidth=1.5, label="Ideal Parity y = x")
    ax2.set_title(f"XGBoost Regressor (Test RMSE: {experiments[5]['Official_Test_RMSE']:.1f}, NASA: {experiments[5]['Official_Test_NASA']:.0f})", fontsize=11, fontweight="bold")
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
    ax.set_title("Test Set Prediction Residuals: e = y_pred - y_true", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual Error (Predicted - Actual RUL)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(figures_dir / "08_residual_error_distribution.png")
    plt.close()

    # Figure 9: Error by Degradation Stage
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
    sns.boxplot(data=df_errors, x="Stage", y="Absolute_Error", hue="Stage", palette="Blues", ax=ax1, width=0.45, legend=False)
    ax1.set_title("XGBoost: Absolute Error by Operating Stage", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Absolute Error |y - y_hat|", fontsize=10)

    sns.boxplot(data=df_errors, x="Stage", y="Signed_Error", hue="Stage", palette="vlag", ax=ax2, width=0.45, legend=False)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.0)
    ax2.set_title("XGBoost: Signed Residual by Operating Stage", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Signed Error (y_hat - y)", fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "09_error_by_degradation_stage.png")
    plt.close()

    # Figure 10: Target Capping Sensitivity Curve (Validation Endpoint vs Test)
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=200)
    caps_plot = [r["RUL_Cap"] for r in cap_results if r["RUL_Cap"] <= 150]
    val_rmse_plot = [r["Val_Endpoint_RMSE"] for r in cap_results if r["RUL_Cap"] <= 150]
    test_rmse_plot = [r["Official_Test_RMSE"] for r in cap_results if r["RUL_Cap"] <= 150]

    color1 = "#2ca02c"
    color2 = "#1f77b4"
    ax1.set_xlabel("RUL Target Capping Threshold (cycles)", fontsize=10)
    ax1.set_ylabel("RMSE (cycles)", fontsize=10)
    ax1.plot(caps_plot, val_rmse_plot, marker="o", color=color1, linewidth=2, label="Validation Endpoint RMSE")
    ax1.plot(caps_plot, test_rmse_plot, marker="s", color=color2, linewidth=2, linestyle="--", label="Official Test RMSE")
    ax1.legend(loc="upper right")

    plt.title("Target Capping Sensitivity: Validation-Based Parameter Selection", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "10_target_cap_sensitivity.png")
    plt.close()

    print("\nBenchmark Execution Complete! Summary Table:")
    print(df_exp.to_string(index=False))


if __name__ == "__main__":
    run_all_benchmarks()

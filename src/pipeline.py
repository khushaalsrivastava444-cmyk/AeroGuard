"""Unified execution pipeline for AeroGuard.

Executes data validation, model benchmarking, validation-based model selection,
final test evaluation, conformal uncertainty calibration, and figure generation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.loader import load_fd001_raw
from src.models.run_benchmarks import run_all_benchmarks
from src.models.run_advanced import run_advanced_experiments


def run_full_pipeline(verbose: bool = True) -> None:
    """Execute the end-to-end reproducible machine learning pipeline."""
    if verbose:
        print("\n" + "=" * 70)
        print("  AEROGUARD: REPRODUCIBLE END-TO-END PIPELINE")
        print("  Condition-Based Maintenance & RUL Intelligence")
        print("=" * 70 + "\n")

    # Step 1: Verify Dataset
    raw_dir = project_root / "data" / "raw"
    if verbose:
        print("[1/3] Verifying NASA C-MAPSS FD001 dataset...")
    train_df, test_df, rul_series = load_fd001_raw(raw_dir)
    if verbose:
        print(f"      Verified {len(train_df)} training cycles, {len(test_df)} test cycles, {len(rul_series)} test engines.")

    # Step 2: Model Benchmarking & Validation-Based Selection
    if verbose:
        print("\n[2/3] Running engine-level validation benchmarks & model selection...")
    run_all_benchmarks()

    # Step 3: Conformal Calibration, Anomaly Detection & SHAP Attribution
    if verbose:
        print("\n[3/3] Running conformal calibration, anomaly detection & SHAP analysis...")
    run_advanced_experiments()

    if verbose:
        print("\n" + "=" * 70)
        print("  PIPELINE EXECUTION COMPLETE")
        print("  All models, metrics, and figures regenerated successfully.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    run_full_pipeline()

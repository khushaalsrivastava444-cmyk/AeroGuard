"""Unit tests for engine-level endpoint cross-validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.models.train import build_pipeline, run_all_cycle_cv, run_endpoint_group_cv


@pytest.fixture
def multi_engine_df():
    """Create a minimal 6-engine training set to test GroupKFold endpoint evaluation."""
    data = []
    # 6 engines, lifespans between 40 and 60 cycles
    for eng in range(1, 7):
        n_cycles = 40 + eng * 3
        for t in range(1, n_cycles + 1):
            row = {
                "engine_id": eng,
                "cycle": t,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            # Provide 21 sensor measurements
            for s in range(1, 22):
                # Simulated degradation signal on s_2, s_4, s_11
                row[f"s_{s}"] = float(100.0 + (t / n_cycles) * 50.0 if s in [2, 4, 11] else 100.0)
            data.append(row)
    return pd.DataFrame(data)


def test_endpoint_group_cv_structure(multi_engine_df):
    """Verify endpoint cross-validation outputs and repeatability."""
    pipe = build_pipeline("ridge", alpha=10.0)

    cv_res_1 = run_endpoint_group_cv(
        pipeline=pipe,
        train_df=multi_engine_df,
        include_temporal=False,
        rul_cap=50,
        n_splits=3,
        seed=42,
        min_observed_cycles=15,
    )

    assert "val_endpoint_mae" in cv_res_1
    assert "val_endpoint_rmse" in cv_res_1
    assert "val_endpoint_r2" in cv_res_1
    assert "val_endpoint_nasa" in cv_res_1

    # Verify seed reproducibility
    cv_res_2 = run_endpoint_group_cv(
        pipeline=pipe,
        train_df=multi_engine_df,
        include_temporal=False,
        rul_cap=50,
        n_splits=3,
        seed=42,
        min_observed_cycles=15,
    )
    assert cv_res_1["val_endpoint_rmse"] == pytest.approx(cv_res_2["val_endpoint_rmse"])


def test_endpoint_cv_vs_all_cycle_cv_distinction(multi_engine_df):
    """Verify endpoint CV and all-cycle CV are distinct metrics."""
    pipe = build_pipeline("ridge", alpha=10.0)

    endpoint_res = run_endpoint_group_cv(
        pipeline=pipe,
        train_df=multi_engine_df,
        include_temporal=False,
        rul_cap=50,
        n_splits=3,
        seed=42,
        min_observed_cycles=15,
    )

    # All-cycle CV requires design matrix X and target y
    from src.features.engineer import prepare_feature_matrix
    X, y, _ = prepare_feature_matrix(multi_engine_df, include_temporal=False, rul_cap=50, is_train=True)

    all_cycle_res = run_all_cycle_cv(
        pipeline=pipe,
        X=X,
        y=y,
        engine_ids=multi_engine_df["engine_id"],
        n_splits=3,
    )

    assert "all_cycle_rmse" in all_cycle_res
    assert "val_endpoint_rmse" in endpoint_res
    # Both evaluate valid non-negative RMSE values
    assert endpoint_res["val_endpoint_rmse"] > 0
    assert all_cycle_res["all_cycle_rmse"] > 0

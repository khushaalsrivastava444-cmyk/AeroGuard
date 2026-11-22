"""Unit tests for feature engineering and RUL target calculation."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import ALL_COLUMNS, load_fd001_train
from src.features.engineer import (
    INFORMATIVE_SENSORS,
    SELECTED_FEATURES,
    add_rul_target,
    compute_temporal_features,
    prepare_feature_matrix,
)


@pytest.fixture
def sample_trajectory_df():
    """Create a minimal synthetic multi-engine dataset with known values."""
    data = []
    # Engine 1: 5 cycles, values 10, 20, 30, 40, 50
    for t, val in enumerate([10.0, 20.0, 30.0, 40.0, 50.0], start=1):
        row = {"engine_id": 1, "cycle": t, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
        for s in range(1, 22):
            row[f"s_{s}"] = val if s == 2 else 100.0
        data.append(row)

    # Engine 2: 3 cycles, values 100, 200, 300
    for t, val in enumerate([100.0, 200.0, 300.0], start=1):
        row = {"engine_id": 2, "cycle": t, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
        for s in range(1, 22):
            row[f"s_{s}"] = val if s == 2 else 100.0
        data.append(row)

    return pd.DataFrame(data)


def test_add_rul_target_uncapped(sample_trajectory_df):
    """Verify uncapped RUL is exactly T_i - t."""
    df_rul = add_rul_target(sample_trajectory_df, rul_cap=None)

    # Engine 1 has 5 cycles (T_1 = 5)
    e1_rul = df_rul[df_rul["engine_id"] == 1]["rul"].tolist()
    assert e1_rul == [4, 3, 2, 1, 0]

    # Engine 2 has 3 cycles (T_2 = 3)
    e2_rul = df_rul[df_rul["engine_id"] == 2]["rul"].tolist()
    assert e2_rul == [2, 1, 0]


def test_add_rul_target_capped(sample_trajectory_df):
    """Verify piecewise-linear RUL is correctly clamped at rul_cap."""
    df_rul = add_rul_target(sample_trajectory_df, rul_cap=2)

    e1_rul = df_rul[df_rul["engine_id"] == 1]["rul"].tolist()
    # Uncapped was [4, 3, 2, 1, 0] -> capped at 2 is [2, 2, 2, 1, 0]
    assert e1_rul == [2, 2, 2, 1, 0]

    with pytest.raises(ValueError):
        add_rul_target(sample_trajectory_df, rul_cap=-5)


def test_temporal_features_no_cross_engine_leakage(sample_trajectory_df):
    """Verify that engine 2 rolling statistics are completely independent of engine 1."""
    df_feat = compute_temporal_features(sample_trajectory_df, sensors=["s_2"], windows=[3])

    # Engine 2 cycle 1 must have rolling mean equal to its own initial value (100.0), NOT contaminated by Engine 1
    e2_c1_mean = df_feat[(df_feat["engine_id"] == 2) & (df_feat["cycle"] == 1)]["s_2_roll_mean_3"].iloc[0]
    assert e2_c1_mean == 100.0

    # Engine 1 cycle 3 rolling mean of [10, 20, 30] = 20.0
    e1_c3_mean = df_feat[(df_feat["engine_id"] == 1) & (df_feat["cycle"] == 3)]["s_2_roll_mean_3"].iloc[0]
    assert e1_c3_mean == 20.0

    # Ensure zero NaN values
    assert df_feat.isna().sum().sum() == 0


def test_prepare_feature_matrix_contracts():
    """Verify feature matrix on real C-MAPSS FD001 training data."""
    train_df = load_fd001_train()
    X, y, feat_cols = prepare_feature_matrix(train_df, include_temporal=True, rul_cap=125, is_train=True)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 20631
    assert len(y) == 20631

    # Guarantee forbidden columns do NOT enter X
    assert "engine_id" not in X.columns
    assert "cycle" not in X.columns
    assert "rul" not in X.columns

    # Verify no nulls
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0

    # Verify target bounds
    assert y.min() == 0
    assert y.max() <= 125

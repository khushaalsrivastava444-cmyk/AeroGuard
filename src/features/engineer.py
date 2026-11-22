"""Temporal feature engineering and RUL target calculation for C-MAPSS data.

All windowed operations are causal (backward-looking only) and evaluated strictly
within individual engine groups to prevent lookahead and cross-engine data leakage.
"""

from typing import Final, Optional, Sequence
import numpy as np
import pandas as pd

from src.data.loader import ALL_COLUMNS

# 14 dynamic sensors identified during EDA variance analysis
INFORMATIVE_SENSORS: Final[list[str]] = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
]

INFORMATIVE_SETTINGS: Final[list[str]] = ["setting_1", "setting_2"]
SELECTED_FEATURES: Final[list[str]] = INFORMATIVE_SETTINGS + INFORMATIVE_SENSORS


def add_rul_target(
    df: pd.DataFrame,
    rul_cap: Optional[int] = 125,
) -> pd.DataFrame:
    """Compute run-to-failure Remaining Useful Life (RUL) target for training trajectories.

    Mathematical Formulation:
        For engine i at cycle t with maximum lifetime T_i:
            RUL(i, t) = T_i - t
        If rul_cap is specified:
            RUL_capped(i, t) = min(T_i - t, rul_cap)

    Args:
        df: Input training dataframe containing 'engine_id' and 'cycle'.
        rul_cap: Upper ceiling threshold for piecewise-linear RUL. If None, uncapped.

    Returns:
        pd.DataFrame: A copy of df with an added 'rul' column.
    """
    df_out = df.copy()
    max_cycles = df_out.groupby("engine_id")["cycle"].transform("max")
    uncapped_rul = max_cycles - df_out["cycle"]

    if rul_cap is not None:
        if rul_cap <= 0:
            raise ValueError(f"rul_cap must be a positive integer, got {rul_cap}")
        df_out["rul"] = uncapped_rul.clip(upper=rul_cap)
    else:
        df_out["rul"] = uncapped_rul

    return df_out


def compute_temporal_features(
    df: pd.DataFrame,
    sensors: Sequence[str] = INFORMATIVE_SENSORS,
    windows: Sequence[int] = (5, 15),
) -> pd.DataFrame:
    """Compute strictly backward-looking rolling statistics and differences per engine.

    Engine boundaries are strictly preserved by grouping by 'engine_id'.

    Features engineered per sensor s and window w:
        - Rolling Mean: mu_{t,w} = (1/w) * sum_{k=0}^{w-1} s_{t-k}
        - Rolling Std:  sigma_{t,w} = sqrt( (1/(w-1)) * sum (s_{t-k} - mu)^2 )
        - Lag Difference: Delta_w s_t = s_t - s_{t-w}

    Args:
        df: Input dataframe containing 'engine_id', 'cycle', and sensor channels.
        sensors: Target sensor column names to engineer.
        windows: Rolling window horizons (cycles).

    Returns:
        pd.DataFrame: A copy of df augmented with rolling and delta features.
    """
    df_out = df.copy()

    # Pre-sort to guarantee causal chronological order
    df_out = df_out.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = df_out.groupby("engine_id")

    for w in windows:
        for s in sensors:
            # 1. Rolling Mean (min_periods=1 ensures no NaNs at trajectory inception)
            mean_col = f"{s}_roll_mean_{w}"
            df_out[mean_col] = (
                grouped[s]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # 2. Rolling Standard Deviation (min_periods=2, fill initial point with 0.0)
            std_col = f"{s}_roll_std_{w}"
            df_out[std_col] = (
                grouped[s]
                .rolling(window=w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

            # 3. First Difference / Velocity (s_t - s_{t-w})
            diff_col = f"{s}_diff_{w}"
            diff_series = grouped[s].diff(periods=w)
            # For t < w, diff against initial observation at t=1
            first_obs = grouped[s].transform("first")
            df_out[diff_col] = diff_series.fillna(df_out[s] - first_obs)

    return df_out


def prepare_feature_matrix(
    df: pd.DataFrame,
    include_temporal: bool = True,
    rul_cap: Optional[int] = 125,
    is_train: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series], list[str]]:
    """Build clean feature matrix X and target y ready for model consumption.

    Args:
        df: Raw trajectory dataframe.
        include_temporal: Whether to compute windowed temporal features.
        rul_cap: Target capping threshold for training labels.
        is_train: If True, computes 'rul' column. If False, returns y as None.

    Returns:
        tuple: (X, y, feature_names)
            - X (pd.DataFrame): Design matrix without engine_id/cycle/rul.
            - y (Optional[pd.Series]): Target vector if is_train=True, else None.
            - feature_names (list[str]): Names of feature columns in X.
    """
    processed = df.copy()

    if is_train:
        processed = add_rul_target(processed, rul_cap=rul_cap)
        y = processed["rul"]
    else:
        y = None

    if include_temporal:
        processed = compute_temporal_features(processed, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
        # Drop raw non-informative channels and index columns
        exclude_cols = set(ALL_COLUMNS) - set(SELECTED_FEATURES)
        exclude_cols.update(["engine_id", "cycle", "rul"])
        feature_cols = [c for c in processed.columns if c not in exclude_cols]
    else:
        feature_cols = SELECTED_FEATURES

    X = processed[feature_cols].copy()
    return X, y, feature_cols

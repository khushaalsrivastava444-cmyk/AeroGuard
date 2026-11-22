"""Model construction, engine-level cross-validation, and official test evaluation.

Guarantees engine-level isolation: data from adjacent observations of the same
engine never leak across training and validation folds.
"""

from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.evaluation.metrics import compute_regression_metrics
from src.features.engineer import (
    INFORMATIVE_SENSORS,
    SELECTED_FEATURES,
    compute_temporal_features,
    prepare_feature_matrix,
)


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Constant baseline regressor predicting the empirical median of training targets."""

    def __init__(self) -> None:
        self.median_: float = 0.0

    def fit(self, X: Any, y: np.ndarray) -> "MedianBaseline":
        self.median_ = float(np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.median_, dtype=float)


def build_pipeline(model_name: str, **kwargs: Any) -> Pipeline:
    """Instantiate a model pipeline with appropriate feature scaling where required."""
    name = model_name.lower()

    if name in ("median", "baseline"):
        model = MedianBaseline()
        return Pipeline([("model", model)])

    elif name == "linear":
        model = LinearRegression()
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    elif name == "ridge":
        alpha = kwargs.get("alpha", 10.0)
        model = Ridge(alpha=alpha, random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    elif name in ("rf", "random_forest"):
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 12)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("model", model)])

    elif name in ("xgb", "xgboost"):
        n_estimators = kwargs.get("n_estimators", 120)
        max_depth = kwargs.get("max_depth", 5)
        learning_rate = kwargs.get("learning_rate", 0.05)
        subsample = kwargs.get("subsample", 0.8)
        colsample_bytree = kwargs.get("colsample_bytree", 0.8)
        reg_alpha = kwargs.get("reg_alpha", 1.0)
        reg_lambda = kwargs.get("reg_lambda", 5.0)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("model", model)])

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: median, linear, ridge, rf, xgb")


def run_engine_group_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    engine_ids: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """Execute GroupKFold cross-validation partitioned strictly by engine ID.

    Adjacent time steps for the same engine are never split across train and validation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    mae_list, rmse_list, r2_list, nasa_list = [], [], [], []

    for train_idx, val_idx in gkf.split(X, y, groups=engine_ids):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        metrics = compute_regression_metrics(y_val.to_numpy(), y_val_pred)
        mae_list.append(metrics["mae"])
        rmse_list.append(metrics["rmse"])
        r2_list.append(metrics["r2"])
        nasa_list.append(metrics["nasa_score"])

    return {
        "cv_mae_mean": float(np.mean(mae_list)),
        "cv_mae_std": float(np.std(mae_list)),
        "cv_rmse_mean": float(np.mean(rmse_list)),
        "cv_rmse_std": float(np.std(rmse_list)),
        "cv_r2_mean": float(np.mean(r2_list)),
        "cv_nasa_mean": float(np.mean(nasa_list)),
    }


def evaluate_on_test_fleet(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    rul_series: pd.Series,
    feature_names: list[str],
    include_temporal: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate pipeline on official NASA test engines at their final observed cycles.

    Simulates the exact operational test protocol:
    1. Augments test trajectories with causal historical features.
    2. Slices the terminal observed cycle for each engine i.
    3. Generates predictions y_hat.
    4. Evaluates against true remaining cycles (rul_series).

    Returns:
        tuple: (metrics_dict, y_true_arr, y_pred_arr)
    """
    if include_temporal:
        test_proc = compute_temporal_features(test_df, sensors=INFORMATIVE_SENSORS, windows=(5, 15))
    else:
        test_proc = test_df.copy()

    # Extract final observed cycle per test engine
    final_cycle_indices = test_proc.groupby("engine_id")["cycle"].idxmax()
    test_last_cycles = test_proc.loc[final_cycle_indices].sort_values("engine_id").reset_index(drop=True)

    X_test_last = test_last_cycles[feature_names]
    y_test_pred = pipeline.predict(X_test_last)

    # Postprocessing constraint: RUL cannot physically be negative
    y_test_pred = np.maximum(y_test_pred, 0.0)

    y_test_true = rul_series.sort_index().to_numpy()
    metrics = compute_regression_metrics(y_test_true, y_test_pred)

    return metrics, y_test_true, y_test_pred


def train_and_save_model(
    model_name: str,
    train_df: pd.DataFrame,
    include_temporal: bool = True,
    rul_cap: Optional[int] = 125,
    save_path: Optional[Union[str, Path]] = None,
    **model_kwargs: Any,
) -> tuple[Pipeline, list[str]]:
    """Train complete model pipeline on full training fleet and persist to disk."""
    X_train, y_train, feature_names = prepare_feature_matrix(
        train_df, include_temporal=include_temporal, rul_cap=rul_cap, is_train=True
    )

    pipeline = build_pipeline(model_name, **model_kwargs)
    pipeline.fit(X_train, y_train)

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model_name": model_name,
            "pipeline": pipeline,
            "feature_names": feature_names,
            "rul_cap": rul_cap,
            "include_temporal": include_temporal,
        }
        joblib.dump(artifact, out_path)

    return pipeline, feature_names

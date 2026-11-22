"""Evaluation metrics for Remaining Useful Life (RUL) estimation.

Includes standard regression metrics (MAE, RMSE, R2) and the official NASA
asymmetric prognostics evaluation metric (PHM08 competition metric).
"""

from typing import Union
import numpy as np


def mean_absolute_error(
    y_true: Union[np.ndarray, list[float]],
    y_pred: Union[np.ndarray, list[float]],
) -> float:
    """Mean Absolute Error (MAE): (1/N) * sum(|y_true - y_pred|)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def root_mean_squared_error(
    y_true: Union[np.ndarray, list[float]],
    y_pred: Union[np.ndarray, list[float]],
) -> float:
    """Root Mean Squared Error (RMSE): sqrt( (1/N) * sum((y_true - y_pred)^2) )."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def r2_score(
    y_true: Union[np.ndarray, list[float]],
    y_pred: Union[np.ndarray, list[float]],
) -> float:
    """Coefficient of Determination (R^2): 1 - SS_res / SS_tot."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def nasa_scoring_function(
    y_true: Union[np.ndarray, list[float]],
    y_pred: Union[np.ndarray, list[float]],
) -> float:
    """NASA Prognostics Center of Excellence asymmetric scoring metric.

    Mathematical Formulation:
        Let prediction error error_i = y_pred_i - y_true_i (predicted - actual).
        For each engine i:
            s_i = exp(-error_i / 13) - 1   if error_i < 0  (Early prediction: conservative)
            s_i = exp(error_i / 10) - 1    if error_i >= 0 (Late prediction: dangerous)
        Total Score S = sum_{i=1}^N s_i

    Sign Convention & Operational Interpretation:
        - error_i < 0: The model predicts maintenance is needed *earlier* than actual failure.
          Penalized moderately (divisor 13) because it leads to premature inspection costs.
        - error_i > 0: The model predicts maintenance *later* than actual failure.
          Penalized aggressively (divisor 10) because late prediction leads to catastrophic
          in-flight engine failure.
        - Lower total scores are superior. Score = 0 represents perfect prediction.

    Args:
        y_true: Array-like of actual ground-truth RUL values.
        y_pred: Array-like of predicted RUL values.

    Returns:
        float: Sum of individual engine asymmetric penalty scores.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true {y_t.shape} vs y_pred {y_p.shape}")

    d = y_p - y_t  # predicted - actual
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_regression_metrics(
    y_true: Union[np.ndarray, list[float]],
    y_pred: Union[np.ndarray, list[float]],
) -> dict[str, float]:
    """Compute full evaluation suite: MAE, RMSE, R2, and NASA Score."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nasa_score": nasa_scoring_function(y_true, y_pred),
    }

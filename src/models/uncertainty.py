"""Conformal prediction and uncertainty estimation for AeroGuard.

Implements Split Conformal Prediction to produce mathematically valid,
distribution-free prediction intervals with user-specified finite-sample
coverage guarantees (e.g., 90% or 95%).
"""

from pathlib import Path
from typing import Any, Optional, Union
import joblib
import numpy as np
import pandas as pd


class SplitConformalPredictor:
    """Split Conformal Prediction interval calibrator.

    Mathematical Formulation:
        Given calibration residuals R_i = |y_i - y_hat_i| for i in {1, ..., n}:
        For significance level alpha (e.g. 0.10 for 90% coverage), the conformal
        quantile is defined as:
            q_alpha = Quantile(R, ceil((n + 1) * (1 - alpha)) / n)
        The prediction interval for a new test instance x is:
            C(x) = [ max(0, y_hat - q_alpha), y_hat + q_alpha ]

    Assumptions & Guarantees:
        Requires only that calibration and test errors are exchangeable (drawn
        from the same underlying data generating distribution). Does not assume
        normality, homoscedasticity, or linearity.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.quantile_: float = 0.0
        self.calibration_residuals_: np.ndarray = np.array([])

    def fit(self, y_cal_true: np.ndarray, y_cal_pred: np.ndarray) -> "SplitConformalPredictor":
        """Compute empirical non-conformity distribution from held-out calibration engines."""
        residuals = np.abs(np.asarray(y_cal_true, dtype=float) - np.asarray(y_cal_pred, dtype=float))
        self.calibration_residuals_ = np.sort(residuals)
        n = len(residuals)

        # Conformal finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile_ = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict_interval(
        self,
        y_pred: Union[np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute [lower_bound, upper_bound] for point predictions.

        Returns:
            tuple: (lower_bounds, upper_bounds) with physical lower bound clipped at 0.0.
        """
        y_p = np.asarray(y_pred, dtype=float)
        lower = np.maximum(0.0, y_p - self.quantile_)
        upper = y_p + self.quantile_
        return lower, upper

    def evaluate_coverage(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate empirical coverage and average width on test fleet."""
        lower, upper = self.predict_interval(y_test_pred)
        y_true = np.asarray(y_test_true, dtype=float)

        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))

        return {
            "nominal_coverage": 1.0 - self.alpha,
            "empirical_coverage": empirical_coverage,
            "mean_interval_width": mean_width,
            "conformal_quantile_q": self.quantile_,
        }

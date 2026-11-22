"""Unit tests for evaluation metrics and NASA prognostics score."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_regression_metrics,
    mean_absolute_error,
    nasa_scoring_function,
    r2_score,
    root_mean_squared_error,
)


def test_standard_metrics_known_values():
    """Verify standard metrics against hand-calculated values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0])  # errors: +2, -2, +2, -2

    # MAE = (|2| + |-2| + |2| + |-2|) / 4 = 2.0
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0)

    # RMSE = sqrt( (4 + 4 + 4 + 4) / 4 ) = 2.0
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(2.0)

    # SS_tot = (10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2 = 225 + 25 + 25 + 225 = 500
    # SS_res = 4 * 4 = 16
    # R2 = 1 - 16/500 = 0.968
    assert r2_score(y_true, y_pred) == pytest.approx(0.968)


def test_nasa_scoring_perfect_prediction():
    """Verify zero error yields exactly score zero."""
    y_true = np.array([50.0, 100.0, 150.0])
    y_pred = np.array([50.0, 100.0, 150.0])
    assert nasa_scoring_function(y_true, y_pred) == pytest.approx(0.0)


def test_nasa_scoring_asymmetry_and_known_values():
    """Verify asymmetric penalization of late vs early predictions against analytical values."""
    # Test case 1: Single early prediction (d = -10, predicted < actual)
    y_true_early = np.array([100.0])
    y_pred_early = np.array([90.0])
    # Expected: exp(10/13) - 1 ~= 1.15818
    expected_early = float(np.exp(10.0 / 13.0) - 1.0)
    assert nasa_scoring_function(y_true_early, y_pred_early) == pytest.approx(expected_early, rel=1e-5)

    # Test case 2: Single late prediction (d = +10, predicted > actual)
    y_true_late = np.array([100.0])
    y_pred_late = np.array([110.0])
    # Expected: exp(10/10) - 1 = exp(1) - 1 ~= 1.71828
    expected_late = float(np.exp(1.0) - 1.0)
    assert nasa_scoring_function(y_true_late, y_pred_late) == pytest.approx(expected_late, rel=1e-5)

    # Late prediction MUST have a higher penalty than early prediction of same absolute magnitude
    assert expected_late > expected_early

    # Test case 3: Combined array
    y_true_both = np.array([100.0, 100.0])
    y_pred_both = np.array([90.0, 110.0])
    assert nasa_scoring_function(y_true_both, y_pred_both) == pytest.approx(expected_early + expected_late)


def test_nasa_scoring_shape_mismatch():
    """Verify ValueError is raised on mismatched input dimensions."""
    with pytest.raises(ValueError):
        nasa_scoring_function([1.0, 2.0], [1.0])


def test_compute_regression_metrics_bundle():
    """Verify comprehensive metric dictionary structure."""
    res = compute_regression_metrics([10.0, 20.0], [11.0, 19.0])
    assert set(res.keys()) == {"mae", "rmse", "r2", "nasa_score"}
    assert all(isinstance(v, float) for v in res.values())

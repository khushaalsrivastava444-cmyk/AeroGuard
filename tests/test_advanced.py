"""Unit tests for conformal uncertainty, anomaly detection, and SHAP explainability."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explainability.shap_explainer import AeroGuardExplainer
from src.models.anomaly import EngineAnomalyDetector
from src.models.uncertainty import SplitConformalPredictor


def test_split_conformal_predictor():
    """Verify SplitConformalPredictor properties and finite-sample quantile calculation."""
    # Calibration errors
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    y_pred = y_true + np.array([-2.0, 1.0, 3.0, -1.0, 4.0, -5.0, 2.0, -3.0, 1.0, 6.0])

    conformal = SplitConformalPredictor(alpha=0.20)  # 80% nominal coverage
    conformal.fit(y_true, y_pred)

    assert conformal.quantile_ > 0.0

    # Test intervals for a prediction of 50.0
    lower, upper = conformal.predict_interval(np.array([50.0]))
    assert lower[0] < 50.0
    assert upper[0] > 50.0
    assert lower[0] >= 0.0

    # Physical clipping check: prediction of 2.0 with quantile 5.0 must clip lower bound at 0.0
    lower_clip, _ = conformal.predict_interval(np.array([2.0]))
    assert lower_clip[0] == 0.0

    # Coverage verification
    eval_res = conformal.evaluate_coverage(y_true, y_pred)
    assert eval_res["empirical_coverage"] >= 0.80
    assert eval_res["mean_interval_width"] > 0.0


def test_split_conformal_small_sample_edge_case():
    """Verify ValueError is raised when sample size is too small to guarantee nominal coverage."""
    # n = 5, alpha = 0.05 -> k = ceil(6 * 0.95) = 6 > 5
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0, 52.0])

    conformal = SplitConformalPredictor(alpha=0.05)
    with pytest.raises(ValueError, match="too small"):
        conformal.fit(y_true, y_pred)


def test_anomaly_detector_scoring():
    """Verify Isolation Forest anomaly detector assigns higher scores to corrupted outliers."""
    np.random.seed(42)
    # Nominal healthy distribution around 0
    X_nominal = pd.DataFrame(np.random.normal(0, 1, size=(200, 5)), columns=[f"f_{i}" for i in range(5)])
    # Extreme anomalous point
    X_outlier = pd.DataFrame([[15.0, -12.0, 20.0, -18.0, 25.0]], columns=[f"f_{i}" for i in range(5)])

    detector = EngineAnomalyDetector(n_estimators=50, random_state=42)
    detector.fit(X_nominal)

    nom_scores = detector.score_samples(X_nominal)
    outlier_score = detector.score_samples(X_outlier)[0]

    assert 0.0 <= outlier_score <= 1.0
    assert outlier_score > np.percentile(nom_scores, 95), "Outlier must receive high anomaly score"


def test_shap_explainer_efficiency_property():
    """Verify Shapley additive efficiency property: sum(SHAP) + base_value == prediction."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, 4), columns=["a", "b", "c", "d"])
    y = 2.0 * X["a"] - 3.0 * X["b"] + 1.5 * X["c"] + 10.0

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)

    explainer = AeroGuardExplainer(model, feature_names=["a", "b", "c", "d"])
    exp_res = explainer.explain_engine(X.iloc[0])

    # Check efficiency: sum of SHAP values + base_value must equal model prediction
    shap_sum = exp_res["contributions"]["SHAP_Value"].sum()
    expected_pred = exp_res["base_value"] + shap_sum
    actual_pred = float(model.predict(pd.DataFrame([X.iloc[0]]))[0])

    assert expected_pred == pytest.approx(actual_pred, abs=1e-4)

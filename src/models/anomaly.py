"""Anomaly detection module for turbofan sensor signatures using Isolation Forest.

Identifies multivariate deviations in the high-dimensional sensor space.
Note: An anomaly score measures geometric isolation in sensor space,
NOT a calibrated failure probability. It provides complementary health indicators.
"""

from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class EngineAnomalyDetector:
    """Multivariate anomaly detector based on Isolation Forest.

    Mathematical Formulation:
        Isolation trees recursively partition feature space. Anomalous points
        (degraded engine states) isolate in fewer partitions than nominal states:
            s(x, n) = 2^( -E(h(x)) / c(n) )
        where E(h(x)) is the mean path length across trees, and c(n) is the
        average path length of unsuccessful searches in a Binary Search Tree (BST).
        Scores s(x) approaching 1 indicate extreme anomalies; scores below 0.5
        indicate normal operational regimes.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "EngineAnomalyDetector":
        """Fit scaler and Isolation Forest on training feature matrix."""
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Compute continuous anomaly score in [0, 1] where 1 is highly anomalous.

        sklearn's score_samples returns opposite of anomaly score (higher is normal).
        We invert and scale to [0, 1] for intuitive monitoring.
        """
        X_scaled = self.scaler.transform(X[self.feature_names_])
        raw_scores = self.model.score_samples(X_scaled)
        # Invert: -raw_scores is larger for anomalies
        norm_scores = 1.0 / (1.0 + np.exp(raw_scores * 5.0))
        return norm_scores

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary labels: -1 for anomaly, +1 for nominal."""
        X_scaled = self.scaler.transform(X[self.feature_names_])
        return self.model.predict(X_scaled)

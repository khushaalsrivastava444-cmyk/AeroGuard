"""Model explainability module using TreeSHAP for tree ensembles.

Computes additive Shapley feature attributions:
    f(x) = phi_0 + sum_{j=1}^M phi_j(x)
where phi_0 is the base expected value E[f(X)], and phi_j is the marginal
attribution of feature j according to cooperative game theory axioms.
"""

from pathlib import Path
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class AeroGuardExplainer:
    """Wrapper around TreeExplainer for AeroGuard XGBoost models."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        # Extract the underlying estimator if passed inside a Pipeline
        underlying_model = model.named_steps["model"] if hasattr(model, "named_steps") else model
        self.explainer = shap.TreeExplainer(underlying_model)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP Explanation object for a dataframe X."""
        X_df = X[self.feature_names]
        shap_values = self.explainer(X_df)
        return shap_values

    def get_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute mean absolute SHAP value per feature: (1/N) * sum |phi_j|."""
        explanation = self.explain(X)
        mean_abs = np.mean(np.abs(explanation.values), axis=0)
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        return df_imp

    def explain_engine(self, engine_features: pd.Series) -> dict[str, Any]:
        """Generate local attribution for a single engine inspection observation."""
        df_single = pd.DataFrame([engine_features])[self.feature_names]
        explanation = self.explain(df_single)
        exp_0 = explanation[0]

        contributions = pd.DataFrame({
            "Feature": self.feature_names,
            "Value": df_single.iloc[0].values,
            "SHAP_Value": exp_0.values,
        }).sort_values("SHAP_Value", key=abs, ascending=False).reset_index(drop=True)

        return {
            "base_value": float(exp_0.base_values),
            "prediction": float(np.sum(exp_0.values) + exp_0.base_values),
            "contributions": contributions,
            "explanation": exp_0,
        }


def get_global_feature_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Convenience helper to obtain global importance table."""
    explainer = AeroGuardExplainer(model, feature_names)
    return explainer.get_global_importance(X)


def explain_single_prediction(
    model: Any,
    engine_features: pd.Series,
    feature_names: list[str],
) -> dict[str, Any]:
    """Convenience helper for local prediction attribution."""
    explainer = AeroGuardExplainer(model, feature_names)
    return explainer.explain_engine(engine_features)

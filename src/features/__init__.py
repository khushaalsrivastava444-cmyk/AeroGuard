"""Feature engineering and target transformation for AeroGuard."""

from src.features.engineer import (
    INFORMATIVE_SENSORS,
    INFORMATIVE_SETTINGS,
    SELECTED_FEATURES,
    add_rul_target,
    compute_temporal_features,
    prepare_feature_matrix,
)

__all__ = [
    "INFORMATIVE_SENSORS",
    "INFORMATIVE_SETTINGS",
    "SELECTED_FEATURES",
    "add_rul_target",
    "compute_temporal_features",
    "prepare_feature_matrix",
]

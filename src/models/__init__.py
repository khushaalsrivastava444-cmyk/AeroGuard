"""Model training, benchmarking, and cross-validation for AeroGuard."""

from src.models.train import (
    MedianBaseline,
    build_pipeline,
    evaluate_on_test_fleet,
    run_engine_group_cv,
    train_and_save_model,
)

__all__ = [
    "MedianBaseline",
    "build_pipeline",
    "run_engine_group_cv",
    "evaluate_on_test_fleet",
    "train_and_save_model",
]

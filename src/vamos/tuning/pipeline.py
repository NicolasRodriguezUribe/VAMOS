"""Compatibility wrapper for NSGA-II tuning pipeline utilities."""

from .meta.pipeline import TuningPipeline, compute_hyperparameter_importance

__all__ = ["TuningPipeline", "compute_hyperparameter_importance"]

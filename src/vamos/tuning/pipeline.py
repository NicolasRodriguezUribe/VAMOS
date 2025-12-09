"""Compatibility wrapper for NSGA-II tuning pipeline utilities."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.pipeline' is deprecated; use 'vamos.tuning.evolver.pipeline' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .evolver.pipeline import TuningPipeline, compute_hyperparameter_importance

__all__ = ["TuningPipeline", "compute_hyperparameter_importance"]

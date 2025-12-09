"""Compatibility wrapper for meta-tuning problem definition."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.meta_problem' is deprecated; use 'vamos.tuning.evolver.meta_problem' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .evolver.meta_problem import MetaOptimizationProblem

__all__ = ["MetaOptimizationProblem"]

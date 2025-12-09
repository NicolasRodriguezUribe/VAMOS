"""Deprecated meta namespace; use vamos.tuning.evolver instead."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.meta' is deprecated; use 'vamos.tuning.evolver' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vamos.tuning.evolver.meta_problem import MetaOptimizationProblem
from vamos.tuning.evolver.nsga2_meta import MetaNSGAII
from vamos.tuning.evolver.tuner import NSGAIITuner
from vamos.tuning.evolver.pipeline import TuningPipeline, compute_hyperparameter_importance
from vamos.tuning.evolver.nsgaii import build_nsgaii_config_space

__all__ = [
    "MetaOptimizationProblem",
    "MetaNSGAII",
    "NSGAIITuner",
    "TuningPipeline",
    "compute_hyperparameter_importance",
    "build_nsgaii_config_space",
]

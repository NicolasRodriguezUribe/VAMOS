"""
Public-facing API surface for VAMOS.

This module re-exports stable entrypoints intended for library consumers.
Internal/experimental modules (runner, CLI helpers, tuning pipelines, etc.)
should be imported explicitly from their modules instead of via the package root.
"""
from __future__ import annotations

from vamos.foundation.constraints import (
    ConstraintHandlingStrategy,
    ConstraintInfo,
    CVAsObjectiveStrategy,
    EpsilonConstraintStrategy,
    FeasibilityFirstStrategy,
    PenaltyCVStrategy,
    compute_constraint_info,
    get_constraint_strategy,
)
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import (
    ProblemSelection,
    ProblemSpec,
    available_problem_names,
    make_problem_selection,
)

__all__ = [
    # Experiment config
    "ExperimentConfig",
    # Problem registry
    "ProblemSpec",
    "ProblemSelection",
    "available_problem_names",
    "make_problem_selection",
    # Constraints
    "ConstraintInfo",
    "ConstraintHandlingStrategy",
    "FeasibilityFirstStrategy",
    "PenaltyCVStrategy",
    "CVAsObjectiveStrategy",
    "EpsilonConstraintStrategy",
    "compute_constraint_info",
    "get_constraint_strategy",
]

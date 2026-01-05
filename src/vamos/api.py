"""
User-facing API surface for VAMOS.

This module exposes the small set of stable entrypoints most users need:
- Programmatic optimization via `optimize` / `OptimizeConfig`.
- Experiment configuration via `ExperimentConfig`.
- Problem selection helpers.
- Basic diagnostics and objective reduction helpers.

For lower-level control, import from the layered packages:
`vamos.foundation.*`, `vamos.engine.*`, `vamos.experiment.*`, `vamos.ux.*`.
"""

from __future__ import annotations

from vamos.experiment.diagnostics.self_check import run_self_check
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.experiment.optimize import (
    OptimizeConfig,
    OptimizationResult,
    run_optimization,
)
from vamos.foundation.metrics.pareto import pareto_filter
from vamos.foundation.problem.registry import (
    ProblemSelection,
    ProblemSpec,
    available_problem_names,
    make_problem_selection,
)
from vamos.ux.analysis.core_objective_reduction import (
    ObjectiveReductionConfig,
    ObjectiveReducer,
    reduce_objectives,
)

# Unified API - the primary entry point
from vamos.experiment.unified import optimize

# Legacy aliases (kept for backward compatibility)
from vamos.experiment.builder import study
from vamos.experiment.auto import auto_optimize

__all__ = [
    # Primary API
    "optimize",
    "run_optimization",
    "study",
    "auto_optimize",
    "OptimizeConfig",
    "OptimizationResult",
    "pareto_filter",
    "ExperimentConfig",
    "ProblemSelection",
    "ProblemSpec",
    "available_problem_names",
    "make_problem_selection",
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    "run_self_check",
]

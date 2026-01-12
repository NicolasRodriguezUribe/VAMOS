"""
VAMOS package facade.

Import most features from the dedicated facades:
- `vamos.api` for core optimization entrypoints.
- `vamos.engine.api` for algorithm configs.
- `vamos.ux.api` for analysis/visualization helpers.
"""

from __future__ import annotations

from vamos.api import (
    ExperimentConfig,
    ObjectiveReductionConfig,
    ObjectiveReducer,
    OptimizeConfig,
    OptimizationResult,
    ProblemSelection,
    ProblemSpec,
    available_problem_names,
    configure_logging,
    make_problem_selection,
    optimize,
    optimize_many,
    pareto_filter,
    reduce_objectives,
    run_self_check,
    suggest_algorithm,
)
from vamos.foundation.version import get_version as _get_version

__all__ = [
    "__version__",
    # Optimization
    "optimize",
    "optimize_many",
    "OptimizeConfig",
    "OptimizationResult",
    "pareto_filter",
    "ExperimentConfig",
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    "run_self_check",
    "suggest_algorithm",
    "configure_logging",
    "available_problem_names",
    "make_problem_selection",
    "ProblemSelection",
    "ProblemSpec",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        return _get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

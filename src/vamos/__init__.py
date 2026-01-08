"""
VAMOS package facade.

Import most features from the dedicated facades:
- `vamos.api` for core optimization entrypoints.
- `vamos.engine.api` for algorithm configs.
- `vamos.ux.api` for analysis/visualization helpers.
- `vamos.experiment.quick` for quick-start runs.
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
    make_problem_selection,
    optimize,
    pareto_filter,
    reduce_objectives,
    run_optimization,
    run_self_check,
)
from vamos.foundation.version import get_version as _get_version

__all__ = [
    "__version__",
    # Optimization
    "optimize",
    "run_optimization",
    "OptimizeConfig",
    "OptimizationResult",
    "pareto_filter",
    "ExperimentConfig",
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    "run_self_check",
    "available_problem_names",
    "make_problem_selection",
    "ProblemSelection",
    "ProblemSpec",
]


def __getattr__(name: str):
    if name == "__version__":
        return _get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

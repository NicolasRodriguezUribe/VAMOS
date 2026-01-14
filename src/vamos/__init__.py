"""
VAMOS package facade.

Import most features from the dedicated facades:
- `vamos.api` for core optimization entrypoints.
- `vamos.algorithms` for algorithm configs and registry helpers.
- `vamos.ux.api` for analysis/visualization helpers.
"""

from __future__ import annotations

from vamos.api import (
    OptimizationResult,
    available_problem_names,
    configure_logging,
    make_problem_selection,
    optimize,
    run_self_check,
)
from vamos.foundation.version import get_version as _get_version

__all__ = [
    "__version__",
    # Optimization
    "optimize",
    "OptimizationResult",
    "configure_logging",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        return _get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

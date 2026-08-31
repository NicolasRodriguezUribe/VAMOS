"""
VAMOS package facade.

Import most features from the dedicated facades:
- `vamos.api` for core optimization entrypoints.
- `vamos.algorithms` for algorithm configs and registry helpers.
- `vamos.problems` for curated benchmark/real-world problem classes.
- `vamos.ux.api` for analysis/visualization helpers.
"""

from __future__ import annotations

import importlib

from vamos.api import (
    CompatibilityReport,
    EnvironmentalSelectionResult,
    IncompleteRunMetadataError,
    LoadLimits,
    OptimizationResult,
    Problem,
    ReplayReport,
    RunManifest,
    StoredRun,
    StudyResult,
    StudySpec,
    VerificationReport,
    available_problem_names,
    configure_logging,
    create_study,
    load_result,
    load_run,
    load_study,
    make_problem,
    make_problem_selection,
    optimize,
    reproduce,
    run_self_check,
    save_result,
    select_survivors,
    verify_run,
)
from vamos.foundation.version import get_version as _get_version

__all__ = [
    "__version__",
    # Optimization
    "optimize",
    "select_survivors",
    "EnvironmentalSelectionResult",
    "Problem",
    "make_problem",
    "OptimizationResult",
    "StudyResult",
    "configure_logging",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
    "save_result",
    "load_run",
    "load_result",
    "StoredRun",
    "RunManifest",
    "LoadLimits",
    "IncompleteRunMetadataError",
    "CompatibilityReport",
    "ReplayReport",
    "VerificationReport",
    "verify_run",
    "reproduce",
    "StudySpec",
    "create_study",
    "load_study",
    "problems",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        return _get_version()
    if name == "problems":
        return importlib.import_module(".problems", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

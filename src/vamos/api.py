"""
User-facing API surface for VAMOS.

This module exposes the small set of stable entrypoints most users need:
- Programmatic optimization via `optimize`.
- Problem selection helpers.
- Basic diagnostics helpers.

For lower-level control, import from the layered packages:
`vamos.foundation.*`, `vamos.engine.*`, `vamos.experiment.*`, `vamos.ux.*`.
"""

from __future__ import annotations

import logging

from vamos.experiment.diagnostics.self_check import run_self_check
from vamos.foundation.logging import configure_vamos_logging
from vamos.experiment.optimization_result import OptimizationResult
from vamos.foundation.problem.builder import make_problem
from vamos.foundation.problem.registry import (
    available_problem_names,
    make_problem_selection,
)

# Unified API - the primary entry point
from vamos.experiment.unified import optimize


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure a minimal console logger for VAMOS.

    Attaches a ``StreamHandler`` to the ``"vamos"`` logger if no handlers
    are already present.  This is intentionally opt-in: library code
    must never call ``logging.basicConfig()``.

    Parameters
    ----------
    level : int, default ``logging.INFO``
        Logging level for the ``"vamos"`` logger.
    """
    configure_vamos_logging(level=level)


__all__ = [
    # Primary API
    "optimize",
    "make_problem",
    "OptimizationResult",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
    "configure_logging",
]

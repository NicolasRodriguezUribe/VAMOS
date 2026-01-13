"""
User-facing API surface for VAMOS.

This module exposes the small set of stable entrypoints most users need:
- Programmatic optimization via `optimize` / `OptimizeConfig`.
- Problem selection helpers.
- Basic diagnostics helpers.

For lower-level control, import from the layered packages:
`vamos.foundation.*`, `vamos.engine.*`, `vamos.experiment.*`, `vamos.ux.*`.
"""

from __future__ import annotations

import logging

from vamos.experiment.diagnostics.self_check import run_self_check
from vamos.foundation.logging import configure_vamos_logging
from vamos.experiment.optimize import OptimizeConfig, OptimizationResult
from vamos.foundation.problem.registry import (
    available_problem_names,
    make_problem_selection,
)

# Unified API - the primary entry point
from vamos.experiment.unified import optimize, optimize_many


def configure_logging(*, level: int = logging.INFO) -> None:
    """
    Configure a minimal console logger for VAMOS.

    This is intentionally opt-in (library code must not call logging.basicConfig()).
    """
    configure_vamos_logging(level=level)


__all__ = [
    # Primary API
    "optimize",
    "optimize_many",
    "OptimizeConfig",
    "OptimizationResult",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
    "configure_logging",
]

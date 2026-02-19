"""
Execution helpers for running a single algorithm instance.

This keeps the algorithm run loop isolated from persistence/plotting concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Container for the raw algorithm payload plus timing."""

    payload: dict[str, Any]
    elapsed_ms: float


def execute_algorithm(
    algorithm: Any,
    problem: Any,
    termination: Any,
    seed: int,
    eval_strategy: Any = None,
    live_viz: Any = None,
) -> ExecutionResult:
    start = time.perf_counter()
    run_fn = algorithm.run
    kwargs = {
        "problem": problem,
        "termination": termination,
        "seed": seed,
        "eval_strategy": eval_strategy,
        "live_viz": live_viz,
    }
    result = run_fn(**kwargs)
    end = time.perf_counter()
    return ExecutionResult(payload=result, elapsed_ms=(end - start) * 1000.0)


__all__ = ["execute_algorithm", "ExecutionResult"]

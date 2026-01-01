"""
Execution helpers for running a single algorithm instance.

This keeps the algorithm run loop isolated from persistence/plotting concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import inspect


@dataclass
class ExecutionResult:
    """Container for the raw algorithm payload plus timing."""

    payload: dict
    elapsed_ms: float


def execute_algorithm(algorithm, problem, termination, seed: int, eval_backend=None, live_viz=None) -> ExecutionResult:
    start = time.perf_counter()
    run_fn = getattr(algorithm, "run")
    sig = inspect.signature(run_fn)
    kwargs = {"problem": problem, "termination": termination, "seed": seed}
    if "eval_backend" in sig.parameters:
        kwargs["eval_backend"] = eval_backend
    if "live_viz" in sig.parameters:
        kwargs["live_viz"] = live_viz
    result = run_fn(**kwargs)
    end = time.perf_counter()
    return ExecutionResult(payload=result, elapsed_ms=(end - start) * 1000.0)


__all__ = ["execute_algorithm", "ExecutionResult"]

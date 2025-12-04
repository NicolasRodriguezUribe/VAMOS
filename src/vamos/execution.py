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

    payload: dict
    elapsed_ms: float


def execute_algorithm(algorithm, problem, termination, seed: int) -> ExecutionResult:
    start = time.perf_counter()
    result = algorithm.run(problem, termination=termination, seed=seed)
    end = time.perf_counter()
    return ExecutionResult(payload=result, elapsed_ms=(end - start) * 1000.0)


__all__ = ["execute_algorithm", "ExecutionResult"]

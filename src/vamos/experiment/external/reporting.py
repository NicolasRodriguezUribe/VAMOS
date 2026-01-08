from __future__ import annotations

import logging
from typing import Any

import numpy as np


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def print_run_banner(problem, problem_selection, algorithm_label: str, backend_label: str, config) -> None:
    spec = getattr(problem_selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    description = getattr(spec, "description", None) or ""
    _logger().info("%s", "=" * 80)
    _logger().info("%s", config.title)
    _logger().info("%s", "=" * 80)
    _logger().info("Problem: %s", label)
    if description:
        _logger().info("Description: %s", description)
    _logger().info("Decision variables: %s", problem.n_var)
    _logger().info("Objectives: %s", problem.n_obj)
    encoding = getattr(problem, "encoding", None)
    if encoding is None:
        encoding = getattr(spec, "encoding", None)
    if encoding is None:
        encoding = "continuous"
    if encoding:
        _logger().info("Encoding: %s", encoding)
    _logger().info("Algorithm: %s", algorithm_label)
    _logger().info("Backend: %s", backend_label)
    _logger().info("Population size: %s", config.population_size)
    _logger().info("Offspring population size: %s", config.offspring_size())
    _logger().info("Max evaluations: %s", config.max_evaluations)
    _logger().info("%s", "-" * 80)


def build_metrics(
    algorithm_name: str,
    engine_name: str,
    total_time_ms: float,
    evaluations: int,
    F: np.ndarray,
    X: np.ndarray | None = None,
) -> dict[str, Any]:
    spread = None
    if F.size and F.shape[1] >= 1:
        spread = np.ptp(F[:, 0])
    evals_per_sec = evaluations / max(1e-9, total_time_ms / 1000.0)
    return {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evaluations": evaluations,
        "evals_per_sec": evals_per_sec,
        "spread": spread,
        "F": F,
        "X": X,
    }


def print_run_results(metrics: dict[str, Any]) -> None:
    algo = metrics["algorithm"]
    time_ms = metrics["time_ms"]
    hv_info = ""
    hv = metrics.get("hv")
    if hv is not None:
        hv_info = f" | HV: {hv:.6f}"
    _logger().info(
        "%s -> Time: %.2f ms | Eval/s: %.1f%s",
        algo,
        time_ms,
        metrics["evals_per_sec"],
        hv_info,
    )
    spread = metrics.get("spread")
    if spread is not None:
        _logger().info("Objective 1 spread: %.6f", spread)


__all__ = ["build_metrics", "print_run_banner", "print_run_results"]

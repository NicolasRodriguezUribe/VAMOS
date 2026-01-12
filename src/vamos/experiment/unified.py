"""
Unified API for VAMOS optimization.

This module provides a single, powerful entry point that consolidates
problem-based runs, study-style configuration, and auto-parameter defaults
into one flexible function.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, overload

from vamos.experiment.optimize import (
    OptimizeConfig,
    OptimizationResult,
    optimize_config as _optimize_config,
    _optimize_problem,
)
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.types import ProblemProtocol
from vamos.experiment.auto import _resolve_problem, _select_algorithm, _compute_pop_size, _compute_budget


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@overload
def optimize(
    problem: OptimizeConfig,
    *,
    engine: str | None = None,
    verbose: bool = False,
) -> OptimizationResult: ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: int = 42,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, Any] | None = None,
    **algo_kwargs: Any,
) -> OptimizationResult: ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: list[int] | tuple[int, ...],
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, Any] | None = None,
    **algo_kwargs: Any,
) -> list[OptimizationResult]: ...


def optimize(
    problem: str | Any | OptimizeConfig,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str | None = None,
    seed: int | list[int] | tuple[int, ...] = 42,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, Any] | None = None,
    **algo_kwargs: Any,
) -> OptimizationResult | list[OptimizationResult]:
    """
    Unified entry point for VAMOS optimization.

    This function consolidates multiple APIs into a single powerful interface:
    - Accepts problem names (strings), instances, or OptimizeConfig
    - Supports AutoML with algorithm="auto"
    - Handles multi-run studies with seed=[0,1,2,...]
    - Prefer config objects (OptimizeConfig + algorithm config builders); raw dicts are not accepted.

    Args:
        problem: Problem name (e.g., "zdt1"), problem instance, or OptimizeConfig.
        algorithm: Algorithm name or "auto" for automatic selection.
        budget: Maximum function evaluations. Auto-determined if None.
        pop_size: Population size. Auto-determined if None.
        engine: Backend engine ("numpy", "numba", "moocore", "jax").
        seed: Random seed or list of seeds for multi-run mode.
        verbose: Print progress information.
        n_var: Override problem dimension when using a string problem key.
        n_obj: Override objective count when using a string problem key.
        problem_kwargs: Extra kwargs forwarded to problem instantiation for string problems.
        **algo_kwargs: Algorithm-config overrides (validated; unknown keys raise ValueError).

    Returns:
        OptimizationResult for single seed, or list[OptimizationResult] for multiple seeds.

    Examples:
        # AutoML mode - zero config
        >>> result = vamos.optimize("zdt1")

        # Specify algorithm
        >>> result = vamos.optimize("zdt1", algorithm="moead", budget=5000)

        # Multi-seed study
        >>> results = vamos.optimize("zdt1", seed=[0, 1, 2, 3, 4])

        # Full control with OptimizeConfig
        >>> result = vamos.optimize(my_config)
    """
    # Case 1: OptimizeConfig passed directly
    if isinstance(problem, OptimizeConfig):
        return _optimize_config(problem, engine=engine)

    # Case 2: Multi-seed mode
    if isinstance(seed, (list, tuple)):
        return [
            _run_single(problem, algorithm, budget, pop_size, engine, s, verbose, n_var, n_obj, problem_kwargs, algo_kwargs) for s in seed
        ]

    # Case 3: Single run
    return _run_single(problem, algorithm, budget, pop_size, engine, seed, verbose, n_var, n_obj, problem_kwargs, algo_kwargs)


def _run_single(
    problem: str | Any,
    algorithm: str,
    budget: int | None,
    pop_size: int | None,
    engine: str | None,
    seed: int,
    verbose: bool,
    n_var: int | None,
    n_obj: int | None,
    problem_kwargs: Mapping[str, Any] | None,
    algo_kwargs: dict[str, Any],
) -> OptimizationResult:
    """Execute a single optimization run."""
    # Resolve problem
    problem_instance = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, problem_kwargs=problem_kwargs)

    # Extract metadata
    n_var = getattr(problem_instance, "n_var", 10)
    n_obj = getattr(problem_instance, "n_obj", 2)
    encoding = normalize_encoding(getattr(problem_instance, "encoding", "real"))

    # Auto-select algorithm if needed
    if algorithm == "auto":
        algorithm = _select_algorithm(n_obj, encoding)
        if verbose:
            _logger().info("[vamos] Auto-selected algorithm: %s", algorithm)

    # Auto-determine hyperparameters if not specified
    effective_pop_size = pop_size if pop_size else _compute_pop_size(n_var, n_obj)
    effective_budget = budget if budget else _compute_budget(n_var, n_obj)
    effective_engine = engine or "numpy"

    if verbose:
        _logger().info("[vamos] Problem: n_var=%s, n_obj=%s, encoding=%s", n_var, n_obj, encoding)
        _logger().info(
            "[vamos] Config: %s, pop_size=%s, budget=%s",
            algorithm,
            effective_pop_size,
            effective_budget,
        )

    return _optimize_problem(
        problem_instance,
        algorithm=algorithm,
        max_evaluations=effective_budget,
        pop_size=effective_pop_size,
        engine=effective_engine,
        seed=seed,
        **algo_kwargs,
    )


__all__ = ["optimize"]

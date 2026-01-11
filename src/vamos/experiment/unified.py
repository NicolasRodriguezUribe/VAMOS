"""
Unified API for VAMOS optimization.

This module provides a single, powerful entry point that consolidates
problem-based runs, study-style configuration, and auto-parameter defaults
into one flexible function.
"""

from __future__ import annotations

import logging
from typing import Any, overload

from vamos.experiment.optimize import (
    OptimizeConfig,
    OptimizationResult,
    optimize_config as _optimize_config,
    _optimize_problem,
)
from vamos.foundation.problem.types import ProblemProtocol
from vamos.experiment.auto import _resolve_problem, _select_algorithm, _compute_pop_size, _compute_budget
from vamos.experiment.builder import study as _study_builder


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@overload
def optimize(
    problem: OptimizeConfig,
    *,
    engine: str = "numpy",
    verbose: bool = False,
) -> OptimizationResult:
    ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str = "numpy",
    seed: int = 42,
    verbose: bool = False,
    **algo_kwargs: Any,
) -> OptimizationResult:
    ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str = "numpy",
    seed: list[int] | tuple[int, ...],
    verbose: bool = False,
    **algo_kwargs: Any,
) -> list[OptimizationResult]:
    ...


def optimize(
    problem: str | Any | OptimizeConfig,
    *,
    algorithm: str = "auto",
    budget: int | None = None,
    pop_size: int | None = None,
    engine: str = "numpy",
    seed: int | list[int] | tuple[int, ...] = 42,
    verbose: bool = False,
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
        **algo_kwargs: Additional algorithm-specific parameters.

    Returns:
        OptimizationResult for single seed, or list[OptimizationResult] for multiple seeds.

    Examples:
        # AutoML mode - zero config
        >>> result = vamos.optimize("zdt1")

        # Specify algorithm
        >>> result = vamos.optimize("zdt1", algorithm="moead", budget=5000)

        # Multi-seed study
        >>> results = vamos.optimize("zdt1", seeds=[0, 1, 2, 3, 4])

        # Full control with OptimizeConfig
        >>> result = vamos.optimize(my_config)
    """
    # Case 1: OptimizeConfig passed directly
    if isinstance(problem, OptimizeConfig):
        return _optimize_config(problem, engine=engine if engine != "numpy" else None)

    # Case 2: Multi-seed mode
    if isinstance(seed, (list, tuple)):
        return [_run_single(problem, algorithm, budget, pop_size, engine, s, verbose, algo_kwargs) for s in seed]

    # Case 3: Single run
    return _run_single(problem, algorithm, budget, pop_size, engine, seed, verbose, algo_kwargs)


def _run_single(
    problem: str | Any,
    algorithm: str,
    budget: int | None,
    pop_size: int | None,
    engine: str,
    seed: int,
    verbose: bool,
    algo_kwargs: dict[str, Any],
) -> OptimizationResult:
    """Execute a single optimization run."""
    # Resolve problem
    problem_instance = _resolve_problem(problem)

    # Extract metadata
    n_var = getattr(problem_instance, "n_var", 10)
    n_obj = getattr(problem_instance, "n_obj", 2)
    encoding = getattr(problem_instance, "encoding", "real")

    # Auto-select algorithm if needed
    if algorithm == "auto":
        algorithm = _select_algorithm(n_obj, encoding)
        if verbose:
            _logger().info("[vamos] Auto-selected algorithm: %s", algorithm)

    # Auto-determine hyperparameters if not specified
    effective_pop_size = pop_size if pop_size else _compute_pop_size(n_var, n_obj)
    effective_budget = budget if budget else _compute_budget(n_var, n_obj)

    if verbose:
        _logger().info("[vamos] Problem: n_var=%s, n_obj=%s, encoding=%s", n_var, n_obj, encoding)
        _logger().info(
            "[vamos] Config: %s, pop_size=%s, budget=%s",
            algorithm,
            effective_pop_size,
            effective_budget,
        )

    # For string problems, use StudyBuilder
    if isinstance(problem, str):
        builder = _study_builder(problem)
        builder = builder.using(algorithm, pop_size=effective_pop_size, **algo_kwargs)
        builder = builder.engine(engine)
        builder = builder.evaluations(effective_budget)
        builder = builder.seed(seed)
        return builder.run()

    return _optimize_problem(
        problem_instance,
        algorithm=algorithm,
        max_evaluations=effective_budget,
        pop_size=effective_pop_size,
        engine=engine,
        seed=seed,
        **algo_kwargs,
    )


__all__ = ["optimize"]

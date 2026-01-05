"""
Auto-optimization: Zero-config entry point for VAMOS.

Analyzes problem metadata and automatically selects appropriate algorithms
and hyperparameters.
"""
from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np

from vamos.experiment.builder import study, StudyBuilder
from vamos.experiment.optimize import OptimizationResult
from vamos.foundation.problem.registry import make_problem_selection


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def auto_optimize(
    problem: Any,
    *,
    budget: int | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Zero-config optimization that automatically selects algorithm and parameters.
    
    Args:
        problem: Problem instance, class, or string name (e.g., "zdt1").
        budget: Maximum function evaluations. If None, auto-determined.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.
        
    Returns:
        OptimizationResult with X (solutions), F (objectives), and metadata.
        
    Example:
        >>> result = vamos.auto_optimize("zdt1")
        >>> result = vamos.auto_optimize(MyProblem(), budget=10000)
    """
    # Resolve problem
    problem_instance = _resolve_problem(problem)
    
    # Extract metadata
    n_var = getattr(problem_instance, "n_var", 10)
    n_obj = getattr(problem_instance, "n_obj", 2)
    n_constr = getattr(problem_instance, "n_constr", 0)
    encoding = getattr(problem_instance, "encoding", "real")
    
    # Select algorithm
    algorithm = _select_algorithm(n_obj, encoding)
    
    # Determine hyperparameters
    pop_size = _compute_pop_size(n_var, n_obj)
    evaluations = budget if budget else _compute_budget(n_var, n_obj)
    
    if verbose:
        _logger().info(
            "[auto_optimize] Problem: n_var=%s, n_obj=%s, encoding=%s",
            n_var,
            n_obj,
            encoding,
        )
        _logger().info(
            "[auto_optimize] Selected: %s, pop_size=%s, evals=%s",
            algorithm,
            pop_size,
            evaluations,
        )
    
    # Build and run - use original problem input if it was a string
    builder = study(problem)  # study() accepts string or instance
    builder = builder.using(algorithm, pop_size=pop_size)
    builder = builder.evaluations(evaluations)
    builder = builder.seed(seed)
    
    return builder.run()


def _resolve_problem(problem: Any) -> Any:
    """Resolve problem from string, class, or instance."""
    if isinstance(problem, str):
        selection = make_problem_selection(problem)
        return selection.instantiate()
    if isinstance(problem, type):
        return problem()
    return problem


def _select_algorithm(n_obj: int, encoding: str) -> str:
    """Select algorithm based on problem characteristics."""
    if n_obj == 1:
        # Single objective - use NSGA-II for now (could add DE later)
        return "nsgaii"
    elif n_obj == 2:
        return "nsgaii"
    elif n_obj >= 3:
        return "nsgaiii"
    return "nsgaii"


def _compute_pop_size(n_var: int, n_obj: int) -> int:
    """Compute reasonable population size."""
    base = max(50, 10 * n_var)
    # For many-objective, prefer larger populations
    if n_obj >= 3:
        base = max(base, 100)
    return min(base, 200)


def _compute_budget(n_var: int, n_obj: int) -> int:
    """Compute reasonable evaluation budget."""
    base = max(1000, 100 * n_var)
    if n_obj >= 3:
        base = max(base, 5000)
    return min(base, 50000)


__all__ = ["auto_optimize"]

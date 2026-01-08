"""
Auto-optimization helpers for the unified optimize entry point.

These utilities derive sensible defaults (algorithm, population size,
evaluation budget) from lightweight problem metadata.
"""

from __future__ import annotations

from typing import Any

from vamos.foundation.problem.registry import make_problem_selection


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

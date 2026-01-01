"""
Quick-start API for VAMOS - one-liner optimization.

This module provides simplified functions for running multi-objective optimization
with minimal configuration. Perfect for quick experiments, tutorials, and prototyping.

Usage:
    from vamos.experiment.quick import run_nsgaii, run_moead, run_spea2

    # Simplest usage - just problem name
    result = run_nsgaii("zdt1")

    # With basic customization
    result = run_nsgaii("zdt1", max_evaluations=10000, pop_size=100, seed=42)

    # Use the result
    print(f"Found {len(result)} solutions")
    result.plot()           # Quick Pareto plot
    result.summary()        # Print metrics
    best = result.best()    # Get knee point solution
"""

from __future__ import annotations

from typing import Any

from vamos.foundation.problem.types import ProblemProtocol

from .config import (
    build_moead_config,
    build_nsgaii_config,
    build_nsgaiii_config,
    build_smsemoa_config,
    build_spea2_config,
    resolve_problem,
)
from .result import QuickResult
from .run import run_optimization


def run_nsgaii(
    problem: str | ProblemProtocol,
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    seed: int = 42,
    n_var: int | None = None,
    n_obj: int | None = None,
    crossover: str = "sbx",
    crossover_prob: float = 0.9,
    crossover_eta: float = 20.0,
    mutation: str = "pm",
    mutation_prob: str | float = "1/n",
    mutation_eta: float = 20.0,
    engine: str = "numpy",
    # Advanced parameters with sensible defaults
    selection_pressure: int = 2,
    archive_size: int | None = None,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    offspring_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
    **problem_kwargs: Any,
) -> QuickResult:
    """
    Run NSGA-II optimization with full configurability.

    This is the single, unified API for NSGA-II. All parameters have sensible
    defaults, but everything is overridable for advanced use cases.
    """
    prob = resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)
    cfg = build_nsgaii_config(
        pop_size=pop_size,
        crossover=crossover,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation=mutation,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        engine=engine,
        selection_pressure=selection_pressure,
        archive_size=archive_size,
        constraint_mode=constraint_mode,
        track_genealogy=track_genealogy,
        offspring_size=offspring_size,
        result_mode=result_mode,
        archive_type=archive_type,
    )

    result = run_optimization(
        problem=prob,
        algorithm="nsgaii",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
    )

    return QuickResult(
        F=result.F,
        X=result.X,
        problem=prob,
        algorithm="nsgaii",
        n_evaluations=max_evaluations,
        seed=seed,
        _raw=result.data,
    )


def run_moead(
    problem: str | ProblemProtocol,
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    seed: int = 42,
    n_var: int | None = None,
    n_obj: int | None = None,
    neighbor_size: int = 20,
    crossover: str = "sbx",
    crossover_prob: float = 0.9,
    crossover_eta: float = 20.0,
    mutation: str = "pm",
    mutation_prob: str | float = "1/n",
    mutation_eta: float = 20.0,
    engine: str = "numpy",
    delta: float = 0.9,
    replace_limit: int = 2,
    aggregation: str = "tchebycheff",
    # Advanced parameters with sensible defaults
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    archive_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
    **problem_kwargs: Any,
) -> QuickResult:
    """Run MOEA/D optimization with full configurability."""
    prob = resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = build_moead_config(
        pop_size=pop_size,
        neighbor_size=neighbor_size,
        delta=delta,
        replace_limit=replace_limit,
        crossover=crossover,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation=mutation,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        aggregation=aggregation,
        engine=engine,
        constraint_mode=constraint_mode,
        track_genealogy=track_genealogy,
        archive_size=archive_size,
        result_mode=result_mode,
        archive_type=archive_type,
    )

    result = run_optimization(
        problem=prob,
        algorithm="moead",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
    )

    return QuickResult(
        F=result.F,
        X=result.X,
        problem=prob,
        algorithm="moead",
        n_evaluations=max_evaluations,
        seed=seed,
        _raw=result.data,
    )


def run_spea2(
    problem: str | ProblemProtocol,
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    archive_size: int | None = None,
    seed: int = 42,
    n_var: int | None = None,
    n_obj: int | None = None,
    crossover: str = "sbx",
    crossover_prob: float = 0.9,
    crossover_eta: float = 20.0,
    mutation: str = "pm",
    mutation_prob: str | float = "1/n",
    mutation_eta: float = 20.0,
    engine: str = "numpy",
    # Advanced parameters with sensible defaults
    selection_pressure: int = 2,
    k_neighbors: int | None = None,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    result_mode: str = "non_dominated",
    **problem_kwargs: Any,
) -> QuickResult:
    """Run SPEA2 optimization with full configurability."""
    prob = resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = build_spea2_config(
        pop_size=pop_size,
        archive_size=archive_size,
        crossover=crossover,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation=mutation,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        selection_pressure=selection_pressure,
        k_neighbors=k_neighbors,
        engine=engine,
        constraint_mode=constraint_mode,
        track_genealogy=track_genealogy,
        result_mode=result_mode,
    )

    result = run_optimization(
        problem=prob,
        algorithm="spea2",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
    )

    return QuickResult(
        F=result.F,
        X=result.X,
        problem=prob,
        algorithm="spea2",
        n_evaluations=max_evaluations,
        seed=seed,
        _raw=result.data,
    )


def run_smsemoa(
    problem: str | ProblemProtocol,
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    seed: int = 42,
    n_var: int | None = None,
    n_obj: int | None = None,
    crossover: str = "sbx",
    crossover_prob: float = 0.9,
    crossover_eta: float = 20.0,
    mutation: str = "pm",
    mutation_prob: str | float = "1/n",
    mutation_eta: float = 20.0,
    engine: str = "numpy",
    # Advanced parameters with sensible defaults
    selection_pressure: int = 2,
    ref_point_offset: float = 0.1,
    ref_point_adaptive: bool = True,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    archive_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
    **problem_kwargs: Any,
) -> QuickResult:
    """Run SMS-EMOA optimization with full configurability."""
    prob = resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = build_smsemoa_config(
        pop_size=pop_size,
        crossover=crossover,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation=mutation,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        selection_pressure=selection_pressure,
        ref_point_offset=ref_point_offset,
        ref_point_adaptive=ref_point_adaptive,
        engine=engine,
        constraint_mode=constraint_mode,
        track_genealogy=track_genealogy,
        archive_size=archive_size,
        result_mode=result_mode,
        archive_type=archive_type,
    )

    result = run_optimization(
        problem=prob,
        algorithm="smsemoa",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
    )

    return QuickResult(
        F=result.F,
        X=result.X,
        problem=prob,
        algorithm="smsemoa",
        n_evaluations=max_evaluations,
        seed=seed,
        _raw=result.data,
    )


def run_nsgaiii(
    problem: str | ProblemProtocol,
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    seed: int = 42,
    n_var: int | None = None,
    n_obj: int | None = None,
    crossover: str = "sbx",
    crossover_prob: float = 0.9,
    crossover_eta: float = 20.0,
    mutation: str = "pm",
    mutation_prob: str | float = "1/n",
    mutation_eta: float = 20.0,
    engine: str = "numpy",
    # Advanced parameters with sensible defaults
    selection_pressure: int = 2,
    ref_divisions: int | None = None,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    result_mode: str = "non_dominated",
    **problem_kwargs: Any,
) -> QuickResult:
    """Run NSGA-III optimization with full configurability."""
    prob = resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = build_nsgaiii_config(
        pop_size=pop_size,
        crossover=crossover,
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation=mutation,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta,
        selection_pressure=selection_pressure,
        ref_divisions=ref_divisions,
        engine=engine,
        constraint_mode=constraint_mode,
        track_genealogy=track_genealogy,
        result_mode=result_mode,
    )

    result = run_optimization(
        problem=prob,
        algorithm="nsgaiii",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
    )

    return QuickResult(
        F=result.F,
        X=result.X,
        problem=prob,
        algorithm="nsgaiii",
        n_evaluations=max_evaluations,
        seed=seed,
        _raw=result.data,
    )


__all__ = [
    "QuickResult",
    "run_nsgaii",
    "run_moead",
    "run_spea2",
    "run_smsemoa",
    "run_nsgaiii",
]

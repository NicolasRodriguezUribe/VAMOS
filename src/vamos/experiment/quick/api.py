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

from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.foundation.problem.types import ProblemProtocol

from .config import (
    build_moead_config,
    build_nsgaii_config,
    build_nsgaiii_config,
    build_smsemoa_config,
    build_spea2_config,
    resolve_problem,
)
from .io import save_quick_result
from .plot import plot_quick_front
from .run import run_optimization


@dataclass
class QuickResult:
    """
    User-friendly result container with convenience methods.

    Attributes:
        F: Objective values array (n_solutions, n_objectives)
        X: Decision variables array (n_solutions, n_variables), may be None
        problem: The problem instance that was optimized
        algorithm: Name of the algorithm used
        n_evaluations: Number of function evaluations performed
        seed: Random seed used
    """

    F: np.ndarray
    X: np.ndarray | None
    problem: ProblemProtocol
    algorithm: str
    n_evaluations: int
    seed: int
    _raw: dict[str, Any]

    def __len__(self) -> int:
        """Return number of solutions in the Pareto front."""
        return self.F.shape[0]

    def __repr__(self) -> str:
        n_obj = self.F.shape[1] if self.F.ndim == 2 else 1
        return (
            f"QuickResult({len(self)} solutions, {n_obj} objectives, "
            f"algorithm='{self.algorithm}', seed={self.seed})"
        )

    def summary(self) -> None:
        """Print a summary of the optimization results."""
        n_obj = self.F.shape[1] if self.F.ndim == 2 else 1
        print("=== VAMOS Quick Result ===")
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Solutions: {len(self)}")
        print(f"Objectives: {n_obj}")
        print(f"Evaluations: {self.n_evaluations}")
        print(f"Seed: {self.seed}")
        print()
        print("Objective ranges:")
        for i in range(n_obj):
            col = self.F[:, i]
            print(f"  f{i+1}: [{col.min():.6f}, {col.max():.6f}]")

        # Compute hypervolume if possible
        try:
            from vamos.foundation.metrics.hypervolume import compute_hypervolume

            ref_point = self.F.max(axis=0) * 1.1
            hv = compute_hypervolume(self.F, ref_point)
            print(f"\nHypervolume (auto ref): {hv:.6f}")
        except Exception:
            pass

    def plot(
        self,
        show: bool = True,
        title: str | None = None,
        labels: tuple[str, str] | tuple[str, str, str] | None = None,
    ) -> Any:
        """
        Plot the Pareto front (2D or 3D based on number of objectives).

        Args:
            show: Whether to display the plot immediately
            title: Custom title for the plot
            labels: Axis labels tuple

        Returns:
            The matplotlib axes object
        """
        return plot_quick_front(
            F=self.F,
            algorithm=self.algorithm,
            problem=self.problem,
            show=show,
            title=title,
            labels=labels,
        )

    def best(self, method: str = "knee") -> dict[str, Any]:
        """
        Select a single 'best' solution from the Pareto front.

        Args:
            method: Selection method - 'knee' (default), 'min_f1', 'min_f2', 'balanced'

        Returns:
            Dictionary with 'X' (decision vars), 'F' (objectives),
            'index' (position in front)
        """
        if method == "knee":
            # Simple knee point: minimize normalized L1 distance
            F_norm = (self.F - self.F.min(axis=0)) / (np.ptp(self.F, axis=0) + 1e-12)
            idx = int(np.argmin(F_norm.sum(axis=1)))
        elif method == "min_f1":
            idx = int(np.argmin(self.F[:, 0]))
        elif method == "min_f2":
            idx = int(np.argmin(self.F[:, 1]))
        elif method == "balanced":
            # Minimize max normalized objective
            F_norm = (self.F - self.F.min(axis=0)) / (np.ptp(self.F, axis=0) + 1e-12)
            idx = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced"
            )

        return {
            "X": self.X[idx] if self.X is not None else None,
            "F": self.F[idx],
            "index": idx,
        }

    def to_dataframe(self) -> Any:
        """
        Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns for each objective (f1, f2, ...) and optionally
            decision variables (x1, x2, ...).

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            ) from exc

        n_obj = self.F.shape[1]
        data = {f"f{i+1}": self.F[:, i] for i in range(n_obj)}

        if self.X is not None:
            n_var = self.X.shape[1]
            for i in range(n_var):
                data[f"x{i+1}"] = self.X[:, i]

        return pd.DataFrame(data)

    def save(self, path: str) -> None:
        """
        Save results to a directory (CSV files for F, X, and metadata).

        Args:
            path: Directory path to save results
        """
        save_quick_result(
            path,
            F=self.F,
            X=self.X,
            problem=self.problem,
            algorithm=self.algorithm,
            n_evaluations=self.n_evaluations,
            seed=self.seed,
        )


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

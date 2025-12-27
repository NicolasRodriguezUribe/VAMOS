"""
Quick-start API for VAMOS - one-liner optimization.

This module provides simplified functions for running multi-objective optimization
with minimal configuration. Perfect for quick experiments, tutorials, and prototyping.

Usage:
    from vamos.quick import run_nsgaii, run_moead, run_spea2

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
from typing import Any, Literal

import numpy as np

from vamos.engine.algorithm.config import (
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.foundation.core.optimize import OptimizeConfig, optimize
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol


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
        from vamos.ux.visualization import plot_pareto_front_2d, plot_pareto_front_3d

        n_obj = self.F.shape[1] if self.F.ndim == 2 else 1
        default_title = f"{self.algorithm.upper()} on {type(self.problem).__name__}"

        if n_obj == 2:
            return plot_pareto_front_2d(
                self.F,
                title=title or default_title,
                labels=labels,
                show=show,
            )
        elif n_obj == 3:
            return plot_pareto_front_3d(
                self.F,
                title=title or default_title,
                labels=labels,
                show=show,
            )
        else:
            raise ValueError(
                f"Cannot plot {n_obj}-objective front directly. "
                "Use parallel coordinates or reduce objectives first."
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
        import json
        from pathlib import Path

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(out_dir / "FUN.csv", self.F, delimiter=",")
        if self.X is not None:
            np.savetxt(out_dir / "X.csv", self.X, delimiter=",")

        metadata = {
            "algorithm": self.algorithm,
            "n_evaluations": self.n_evaluations,
            "seed": self.seed,
            "n_solutions": len(self),
            "n_objectives": self.F.shape[1],
            "problem": type(self.problem).__name__,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {out_dir}")


def _resolve_problem(
    problem: str | ProblemProtocol,
    n_var: int | None = None,
    n_obj: int | None = None,
    **problem_kwargs: Any,
) -> ProblemProtocol:
    """Resolve problem from string name or return as-is if already a Problem."""
    if isinstance(problem, str):
        kwargs = dict(problem_kwargs)
        if n_var is not None:
            kwargs["n_var"] = n_var
        if n_obj is not None:
            kwargs["n_obj"] = n_obj
        selection = make_problem_selection(problem, **kwargs)
        return selection.instantiate()
    return problem


def _build_nsgaii_config(
    pop_size: int,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    engine: str,
    selection_pressure: int = 2,
    archive_size: int | None = None,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    offspring_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
) -> dict[str, Any]:
    """Build NSGA-II config dict with all available parameters."""
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .survival("nsga2")
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if offspring_size is not None:
        cfg = cfg.offspring_size(offspring_size)
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    return cfg.fixed().to_dict()


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

    Args:
        problem: Problem name (e.g., 'zdt1', 'dtlz2') or Problem instance
        max_evaluations: Maximum number of function evaluations
        pop_size: Population size
        seed: Random seed for reproducibility
        n_var: Number of decision variables (for benchmark problems)
        n_obj: Number of objectives (for scalable problems like DTLZ)
        crossover: Crossover operator ('sbx', 'blx_alpha', etc.)
        crossover_prob: Crossover probability
        crossover_eta: Crossover distribution index
        mutation: Mutation operator ('pm', 'polynomial', etc.)
        mutation_prob: Mutation probability ('1/n' or float)
        mutation_eta: Mutation distribution index
        engine: Computation backend ('numpy', 'numba', 'moocore')
        selection_pressure: Tournament selection pressure (default: 2)
        archive_size: External archive size (None = no archive)
        constraint_mode: How to handle constraints ('feasibility', 'penalty', 'none')
        track_genealogy: Whether to track parent-child relationships
        offspring_size: Number of offspring per generation (default: pop_size)
        result_mode: What to return ('non_dominated', 'population', 'external_archive')
        archive_type: Archive pruning strategy ('hypervolume', 'crowding')
        **problem_kwargs: Additional problem-specific parameters

    Returns:
        QuickResult with Pareto front and convenience methods

    Examples:
        # Minimal usage - all defaults
        >>> result = run_nsgaii("zdt1")

        # Basic customization
        >>> result = run_nsgaii("zdt1", max_evaluations=20000, pop_size=200)

        # Advanced configuration - same function!
        >>> result = run_nsgaii(
        ...     "zdt1",
        ...     selection_pressure=4,
        ...     archive_size=100,
        ...     crossover_prob=0.95,
        ...     track_genealogy=True,
        ... )
    """
    prob = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)
    cfg = _build_nsgaii_config(
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

    result = optimize(
        OptimizeConfig(
            problem=prob,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
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
    """
    Run MOEA/D optimization with full configurability.

    This is the single, unified API for MOEA/D. All parameters have sensible
    defaults, but everything is overridable for advanced use cases.

    Args:
        problem: Problem name or Problem instance
        max_evaluations: Maximum function evaluations
        pop_size: Population size (number of weight vectors)
        seed: Random seed
        n_var: Number of decision variables
        n_obj: Number of objectives
        neighbor_size: Size of neighborhood for mating
        crossover: Crossover operator
        crossover_prob: Crossover probability
        crossover_eta: Crossover distribution index
        mutation: Mutation operator
        mutation_prob: Mutation probability
        mutation_eta: Mutation distribution index
        engine: Computation backend
        delta: Probability of selecting from neighborhood vs population
        replace_limit: Maximum number of solutions to replace per iteration
        aggregation: Aggregation function ('tchebycheff', 'weighted_sum', 'pbi')
        constraint_mode: How to handle constraints ('feasibility', 'penalty', 'none')
        track_genealogy: Whether to track parent-child relationships
        archive_size: External archive size (None = no archive)
        result_mode: What to return ('non_dominated', 'population', 'external_archive')
        archive_type: Archive pruning strategy ('hypervolume', 'crowding')
        **problem_kwargs: Problem-specific parameters

    Returns:
        QuickResult with Pareto front and convenience methods

    Examples:
        # Minimal usage - all defaults
        >>> result = run_moead("zdt1")

        # Basic customization
        >>> result = run_moead("zdt1", aggregation="pbi", neighbor_size=30)

        # Advanced configuration - same function!
        >>> result = run_moead("zdt1", archive_size=100, track_genealogy=True)
    """
    prob = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = (
        MOEADConfig()
        .pop_size(pop_size)
        .neighbor_size(neighbor_size)
        .delta(delta)
        .replace_limit(replace_limit)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .aggregation(aggregation)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    cfg = cfg.fixed()

    result = optimize(
        OptimizeConfig(
            problem=prob,
            algorithm="moead",
            algorithm_config=cfg.to_dict(),
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
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
    """
    Run SPEA2 optimization with full configurability.

    This is the single, unified API for SPEA2. All parameters have sensible
    defaults, but everything is overridable for advanced use cases.

    Args:
        problem: Problem name or Problem instance
        max_evaluations: Maximum function evaluations
        pop_size: Population size
        archive_size: Archive size (defaults to pop_size)
        seed: Random seed
        n_var: Number of decision variables
        n_obj: Number of objectives
        crossover: Crossover operator
        crossover_prob: Crossover probability
        crossover_eta: Crossover distribution index
        mutation: Mutation operator
        mutation_prob: Mutation probability
        mutation_eta: Mutation distribution index
        engine: Computation backend
        selection_pressure: Tournament selection pressure (default: 2)
        k_neighbors: Number of neighbors for density estimation (None = sqrt(pop))
        constraint_mode: How to handle constraints ('feasibility', 'penalty', 'none')
        track_genealogy: Whether to track parent-child relationships
        result_mode: What to return ('non_dominated', 'population')
        **problem_kwargs: Problem-specific parameters

    Returns:
        QuickResult with Pareto front and convenience methods

    Examples:
        # Minimal usage - all defaults
        >>> result = run_spea2("zdt1")

        # Basic customization
        >>> result = run_spea2("zdt1", archive_size=200)

        # Advanced configuration - same function!
        >>> result = run_spea2("zdt1", selection_pressure=4, track_genealogy=True)
    """
    prob = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = (
        SPEA2Config()
        .pop_size(pop_size)
        .archive_size(archive_size or pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
    )
    if k_neighbors is not None:
        cfg = cfg.k_neighbors(k_neighbors)
    cfg = cfg.fixed()

    result = optimize(
        OptimizeConfig(
            problem=prob,
            algorithm="spea2",
            algorithm_config=cfg.to_dict(),
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
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
    """
    Run SMS-EMOA optimization with full configurability.

    This is the single, unified API for SMS-EMOA. All parameters have sensible
    defaults, but everything is overridable for advanced use cases.

    Args:
        problem: Problem name or Problem instance
        max_evaluations: Maximum function evaluations
        pop_size: Population size
        seed: Random seed
        n_var: Number of decision variables
        n_obj: Number of objectives
        crossover: Crossover operator
        crossover_prob: Crossover probability
        crossover_eta: Crossover distribution index
        mutation: Mutation operator
        mutation_prob: Mutation probability
        mutation_eta: Mutation distribution index
        engine: Computation backend
        selection_pressure: Tournament selection pressure (default: 2)
        ref_point_offset: Offset for adaptive reference point (default: 0.1)
        ref_point_adaptive: Whether to use adaptive reference point (default: True)
        constraint_mode: How to handle constraints ('feasibility', 'penalty', 'none')
        track_genealogy: Whether to track parent-child relationships
        archive_size: External archive size (None = no archive)
        result_mode: What to return ('non_dominated', 'population', 'external_archive')
        archive_type: Archive pruning strategy ('hypervolume', 'crowding')
        **problem_kwargs: Problem-specific parameters

    Returns:
        QuickResult with Pareto front and convenience methods

    Examples:
        # Minimal usage - all defaults
        >>> result = run_smsemoa("zdt1")

        # Basic customization
        >>> result = run_smsemoa("zdt1", pop_size=50)

        # Advanced configuration - same function!
        >>> result = run_smsemoa("zdt1", archive_size=100, ref_point_offset=0.2)
    """
    prob = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = (
        SMSEMOAConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .reference_point(offset=ref_point_offset, adaptive=ref_point_adaptive)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    cfg = cfg.fixed()

    result = optimize(
        OptimizeConfig(
            problem=prob,
            algorithm="smsemoa",
            algorithm_config=cfg.to_dict(),
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
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
    """
    Run NSGA-III optimization with full configurability.

    This is the single, unified API for NSGA-III. All parameters have sensible
    defaults, but everything is overridable for advanced use cases.
    Recommended for many-objective optimization (3+ objectives).

    Args:
        problem: Problem name or Problem instance
        max_evaluations: Maximum function evaluations
        pop_size: Population size
        seed: Random seed
        n_var: Number of decision variables
        n_obj: Number of objectives
        crossover: Crossover operator
        crossover_prob: Crossover probability
        crossover_eta: Crossover distribution index
        mutation: Mutation operator
        mutation_prob: Mutation probability
        mutation_eta: Mutation distribution index
        engine: Computation backend
        selection_pressure: Tournament selection pressure (default: 2)
        ref_divisions: Number of divisions for reference directions (auto if None)
        constraint_mode: How to handle constraints ('feasibility', 'penalty', 'none')
        track_genealogy: Whether to track parent-child relationships
        result_mode: What to return ('non_dominated', 'population')
        **problem_kwargs: Problem-specific parameters

    Returns:
        QuickResult with Pareto front and convenience methods

    Examples:
        # Minimal usage - all defaults
        >>> result = run_nsgaiii("dtlz2", n_obj=3)

        # Basic customization
        >>> result = run_nsgaiii("dtlz2", n_obj=3, ref_divisions=12)

        # Advanced configuration - same function!
        >>> result = run_nsgaiii("dtlz2", n_obj=5, selection_pressure=4, track_genealogy=True)
    """
    prob = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, **problem_kwargs)

    cfg = (
        NSGAIIIConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
    )
    if ref_divisions is not None:
        cfg = cfg.reference_directions(divisions=ref_divisions)
    cfg = cfg.fixed()

    result = optimize(
        OptimizeConfig(
            problem=prob,
            algorithm="nsgaiii",
            algorithm_config=cfg.to_dict(),
            termination=("n_eval", max_evaluations),
            seed=seed,
            engine=engine,
        )
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


AlgorithmName = Literal[
    "nsgaii", "moead", "spea2", "smsemoa", "nsgaiii", "ibea", "smpso"
]



__all__ = [
    "QuickResult",
    "run_nsgaii",
    "run_moead",
    "run_spea2",
    "run_smsemoa",
    "run_nsgaiii",
]

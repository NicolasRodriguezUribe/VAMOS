"""
Experiment context manager for VAMOS.

Provides a clean way to run experiments with automatic resource management,
timing, and result collection.

Example:
    from vamos.experiment import Experiment
    from vamos.foundation.problems_registry import ZDT1

    with Experiment("my_study", output_dir="results") as exp:
        result = exp.optimize(ZDT1(n_var=30), "nsgaii", max_evaluations=5000)
        exp.optimize(ZDT1(n_var=30), "moead", max_evaluations=5000)

    print(exp.summary())  # Shows all results with timing
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from vamos.experiment.optimize import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record of a single optimization run."""

    name: str
    algorithm: str
    problem_name: str
    n_solutions: int
    n_objectives: int
    elapsed_seconds: float
    result: "OptimizationResult"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Summary of all runs in an experiment."""

    name: str
    total_runs: int
    total_time_seconds: float
    runs: list[RunRecord]

    def __str__(self) -> str:
        lines = [
            f"=== Experiment: {self.name} ===",
            f"Total runs: {self.total_runs}",
            f"Total time: {self.total_time_seconds:.2f}s",
            "",
            "Runs:",
        ]
        for i, run in enumerate(self.runs, 1):
            lines.append(f"  {i}. {run.algorithm} on {run.problem_name}: {run.n_solutions} solutions, {run.elapsed_seconds:.2f}s")
        return "\n".join(lines)


class Experiment:
    """
    Context manager for running optimization experiments.

    Provides automatic timing, result collection, and optional output management.

    Args:
        name: Name for this experiment (used in output paths)
        output_dir: Directory for saving results (None = don't save)
        seed: Base random seed (incremented for each run)
        verbose: Print progress messages

    Examples:
        Basic usage:
            with Experiment("study1") as exp:
                result = exp.optimize(ZDT1(n_var=30), "nsgaii")

        Multiple algorithms:
            with Experiment("comparison", output_dir="results") as exp:
                for algo in ["nsgaii", "moead", "spea2"]:
                    exp.optimize(problem, algo, max_evaluations=10000)
            print(exp.summary())

        Access results after:
            with Experiment("test") as exp:
                exp.optimize(problem, "nsgaii")
            best = exp.results[0].best("knee")
    """

    def __init__(
        self,
        name: str = "experiment",
        *,
        output_dir: str | Path | None = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else None
        self.base_seed = seed
        self.verbose = verbose

        self._runs: list[RunRecord] = []
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._run_counter = 0
        self._active = False

    def __enter__(self) -> "Experiment":
        """Start the experiment session."""
        self._active = True
        self._start_time = time.perf_counter()
        if self.verbose:
            logger.info("Starting experiment: %s", self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the experiment session."""
        self._end_time = time.perf_counter()
        self._active = False

        if self.verbose:
            elapsed = self._end_time - (self._start_time or self._end_time)
            logger.info(
                "Experiment '%s' completed: %s runs in %.2fs",
                self.name,
                len(self._runs),
                elapsed,
            )

        # Don't suppress exceptions
        return None

    def optimize(
        self,
        problem: Any,
        algorithm: str = "nsgaii",
        *,
        max_evaluations: int = 10000,
        pop_size: int = 100,
        engine: str = "numpy",
        seed: int | None = None,
        name: str | None = None,
        save: bool = True,
        **kwargs: Any,
    ) -> "OptimizationResult":
        """
        Run an optimization and record the result.

        Args:
            problem: Problem instance to optimize
            algorithm: Algorithm name
            max_evaluations: Maximum function evaluations
            pop_size: Population size
            engine: Backend engine
            seed: Random seed (default: auto-increment from base_seed)
            name: Custom name for this run
            save: Whether to save results to output_dir
            **kwargs: Additional algorithm parameters

        Returns:
            OptimizationResult with Pareto front
        """
        from vamos.experiment.optimize import run_optimization

        if not self._active:
            raise RuntimeError("Experiment not active. Use 'with Experiment(...) as exp:' context.")

        # Determine seed
        if seed is None:
            seed = self.base_seed + self._run_counter

        # Get problem name
        problem_name = getattr(problem, "name", None) or type(problem).__name__

        # Run name
        run_name = name or f"{algorithm}_{problem_name}_{self._run_counter}"

        if self.verbose:
            logger.info("Running: %s on %s...", algorithm, problem_name)

        # Time the run
        start = time.perf_counter()
        result = run_optimization(
            problem,
            algorithm,
            max_evaluations=max_evaluations,
            pop_size=pop_size,
            engine=engine,
            seed=seed,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        if self.verbose:
            logger.info("%s solutions in %.2fs", result.F.shape[0], elapsed)

        # Record the run
        record = RunRecord(
            name=run_name,
            algorithm=algorithm,
            problem_name=problem_name,
            n_solutions=result.F.shape[0],
            n_objectives=result.F.shape[1],
            elapsed_seconds=elapsed,
            result=result,
            metadata={
                "max_evaluations": max_evaluations,
                "pop_size": pop_size,
                "engine": engine,
                "seed": seed,
            },
        )
        self._runs.append(record)
        self._run_counter += 1

        # Save if requested
        if save and self.output_dir:
            out_path = self.output_dir / self.name / run_name
            result.save(str(out_path))

        return result

    @property
    def results(self) -> list["OptimizationResult"]:
        """Get list of all optimization results."""
        return [run.result for run in self._runs]

    @property
    def runs(self) -> list[RunRecord]:
        """Get list of all run records with metadata."""
        return self._runs

    def summary(self) -> ExperimentSummary:
        """Get experiment summary."""
        total_time = sum(r.elapsed_seconds for r in self._runs)
        return ExperimentSummary(
            name=self.name,
            total_runs=len(self._runs),
            total_time_seconds=total_time,
            runs=self._runs,
        )

    def best_run(self, metric: str = "n_solutions") -> RunRecord | None:
        """
        Get the best run by a metric.

        Args:
            metric: 'n_solutions' (most), 'time' (fastest), or 'hv' (if available)

        Returns:
            Best RunRecord or None if no runs
        """
        if not self._runs:
            return None

        if metric == "n_solutions":
            return max(self._runs, key=lambda r: r.n_solutions)
        elif metric == "time":
            return min(self._runs, key=lambda r: r.elapsed_seconds)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use: n_solutions, time")

    def compare(self) -> dict[str, Any]:
        """
        Compare all runs and return comparison data.

        Returns:
            Dict with algorithm names as keys, stats as values
        """
        comparison = {}
        for run in self._runs:
            key = f"{run.algorithm}_{run.problem_name}"
            comparison[key] = {
                "algorithm": run.algorithm,
                "problem": run.problem_name,
                "n_solutions": run.n_solutions,
                "time": run.elapsed_seconds,
                "f_min": run.result.F.min(axis=0).tolist(),
                "f_max": run.result.F.max(axis=0).tolist(),
            }
        return comparison

    def to_dataframe(self):
        """
        Convert runs to a pandas DataFrame.

        Requires pandas to be installed.

        Returns:
            DataFrame with run information
        """
        try:
            import pandas as pd
        except ImportError:
            from vamos.exceptions import DependencyError

            raise DependencyError("pandas", "to_dataframe()", "pip install pandas")

        data = []
        for run in self._runs:
            row = {
                "name": run.name,
                "algorithm": run.algorithm,
                "problem": run.problem_name,
                "n_solutions": run.n_solutions,
                "n_objectives": run.n_objectives,
                "time_seconds": run.elapsed_seconds,
                **run.metadata,
            }
            data.append(row)
        return pd.DataFrame(data)


@contextmanager
def experiment(
    name: str = "experiment",
    *,
    output_dir: str | Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> Iterator[Experiment]:
    """
    Functional context manager for experiments.

    Equivalent to `with Experiment(...) as exp:` but as a function.

    Example:
        from vamos.experiment import experiment
        from vamos.foundation.problems_registry import ZDT1

        with experiment("study", output_dir="results") as exp:
            exp.optimize(ZDT1(n_var=30), "nsgaii")
    """
    exp = Experiment(name, output_dir=output_dir, seed=seed, verbose=verbose)
    with exp:
        yield exp


__all__ = ["Experiment", "experiment", "RunRecord", "ExperimentSummary"]

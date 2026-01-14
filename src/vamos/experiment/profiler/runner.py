"""
Performance profiling runner for comparing backends.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from collections.abc import Sequence

from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result from a single profiling run."""

    engine: str
    time_seconds: float
    n_solutions: int
    hypervolume: float | None = None
    memory_mb: float | None = None


@dataclass
class ProfileReport:
    """Aggregated profiling report."""

    problem: str
    budget: int
    results: list[ProfileResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print formatted summary table."""
        _logger().info("%s", "=" * 50)
        _logger().info("VAMOS Performance Profile")
        _logger().info("%s", "=" * 50)
        _logger().info("Problem: %s", self.problem)
        _logger().info("Budget: %s evaluations", self.budget)

        # Header
        _logger().info("%-12s %-10s %-10s %-10s", "Engine", "Time (s)", "Solutions", "HV")
        _logger().info("%s", "-" * 42)

        # Results
        baseline_time = None
        for r in self.results:
            if baseline_time is None:
                baseline_time = r.time_seconds
            hv_str = f"{r.hypervolume:.4f}" if r.hypervolume else "N/A"
            _logger().info(
                "%-12s %-10.3f %-10s %-10s",
                r.engine,
                r.time_seconds,
                r.n_solutions,
                hv_str,
            )

        # Speedups
        if baseline_time and len(self.results) > 1:
            _logger().info("Speedup vs %s:", self.results[0].engine)
            for r in self.results[1:]:
                if r.time_seconds > 0:
                    speedup = baseline_time / r.time_seconds
                    _logger().info("  %s: %.1fx", r.engine, speedup)

    def to_csv(self, path: str) -> None:
        """Export results to CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "time_seconds", "n_solutions", "hypervolume"])
            for r in self.results:
                writer.writerow([r.engine, r.time_seconds, r.n_solutions, r.hypervolume])


def run_profile(
    problem: str,
    engines: Sequence[str] = ("numpy",),
    budget: int = 2000,
    seed: int = 42,
    compute_hv: bool = True,
) -> ProfileReport:
    """
    Profile optimization across multiple backends.

    Args:
        problem: Problem name (e.g., "zdt1")
        engines: List of engines to compare
        budget: Evaluation budget per run
        seed: Random seed for reproducibility
        compute_hv: Whether to compute hypervolume

    Returns:
        ProfileReport with timing and quality metrics
    """
    report = ProfileReport(problem=problem, budget=budget)

    # Get reference point for HV
    ref_point: list[float] | None = None
    if compute_hv:
        try:
            selection = make_problem_selection(problem)
            prob = selection.instantiate()
            n_obj = getattr(prob, "n_obj", 2)
            ref_point = [1.1] * n_obj  # Standard reference point
        except Exception:
            compute_hv = False

    for engine in engines:
        _logger().info("Profiling %s...", engine)

        try:
            start = time.perf_counter()
            result = optimize(problem, engine=engine, budget=budget, seed=seed, verbose=False)
            elapsed = time.perf_counter() - start

            n_solutions = len(result) if result else 0

            # Compute hypervolume if requested
            hv = None
            if compute_hv and ref_point is not None and result.F is not None and len(result.F) > 0:
                try:
                    from vamos.foundation.metrics.hypervolume import compute_hypervolume

                    hv = compute_hypervolume(result.F, ref_point)
                except Exception:
                    pass

            report.results.append(ProfileResult(engine=engine, time_seconds=elapsed, n_solutions=n_solutions, hypervolume=hv))

        except Exception as exc:
            _logger().warning("  ERROR: %s", exc)
            report.results.append(ProfileResult(engine=engine, time_seconds=0, n_solutions=0, hypervolume=None))

    return report


__all__ = ["run_profile", "ProfileResult", "ProfileReport"]

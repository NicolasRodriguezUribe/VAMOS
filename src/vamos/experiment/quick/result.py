from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.foundation.problem.types import ProblemProtocol

from .io import save_quick_result
from .plot import plot_quick_front

logger = logging.getLogger(__name__)


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
        return f"QuickResult({len(self)} solutions, {n_obj} objectives, algorithm='{self.algorithm}', seed={self.seed})"

    def summary(self) -> None:
        """Print a summary of the optimization results."""
        n_obj = self.F.shape[1] if self.F.ndim == 2 else 1
        logger.info("=== VAMOS Quick Result ===")
        logger.info("Algorithm: %s", self.algorithm.upper())
        logger.info("Solutions: %s", len(self))
        logger.info("Objectives: %s", n_obj)
        logger.info("Evaluations: %s", self.n_evaluations)
        logger.info("Seed: %s", self.seed)
        logger.info("Objective ranges:")
        for i in range(n_obj):
            col = self.F[:, i]
            logger.info("  f%s: [%.6f, %.6f]", i + 1, col.min(), col.max())

        # Compute hypervolume if possible
        try:
            from vamos.foundation.metrics.hypervolume import compute_hypervolume

            ref_point = self.F.max(axis=0) * 1.1
            hv = compute_hypervolume(self.F, ref_point)
            logger.info("Hypervolume (auto ref): %.6f", hv)
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
            raise ValueError(f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced")

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
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas") from exc

        n_obj = self.F.shape[1]
        data = {f"f{i + 1}": self.F[:, i] for i in range(n_obj)}

        if self.X is not None:
            n_var = self.X.shape[1]
            for i in range(n_var):
                data[f"x{i + 1}"] = self.X[:, i]

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

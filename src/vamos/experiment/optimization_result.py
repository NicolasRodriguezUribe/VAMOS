from __future__ import annotations

import logging
from typing import Any, Literal, Mapping, overload

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.metrics.pareto import pareto_filter


def _logger() -> logging.Logger:
    return logging.getLogger("vamos.experiment.optimize")


class OptimizationResult:
    """
    Container returned by optimize() with user-friendly helper methods.

    Attributes:
        F: Objective values array (n_solutions, n_objectives)
        X: Decision variables array (n_solutions, n_variables), may be None
        data: Full result dictionary with all fields

    Examples:
        >>> result = optimize(config)
        >>> result.summary()  # Print quick overview
        >>> result.plot()  # Visualize Pareto front
        >>> best = result.best("knee")  # Select a solution
        >>> df = result.to_dataframe()  # Export to pandas
    """

    def __init__(self, payload: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None):
        self.F: NDArray[Any] | None = payload.get("F")
        self.X: NDArray[Any] | None = payload.get("X")
        self.data: dict[str, Any] = dict(payload)
        self.meta: dict[str, Any] = dict(meta or {})

    def __len__(self) -> int:
        """Number of solutions in the result."""
        return len(self.F) if self.F is not None else 0

    def __repr__(self) -> str:
        n_sol = len(self)
        n_obj = self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0
        return f"OptimizationResult({n_sol} solutions, {n_obj} objectives)"

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0

    def summary_text(self) -> str:
        """Return a human-readable summary string (no logging side effects)."""
        algo = self.meta.get("algorithm")
        seed = self.meta.get("seed")
        engine = self.meta.get("engine")

        lines = [
            "=== Optimization Result ===",
            *([f"Algorithm: {algo}"] if algo else []),
            *([f"Engine: {engine}"] if engine else []),
            *([f"Seed: {seed}"] if seed is not None else []),
            f"Solutions: {len(self)}",
            f"Objectives: {self.n_objectives}",
        ]

        if self.F is not None and len(self.F) > 0:
            lines.append("Objective ranges:")
            for i in range(self.n_objectives):
                col = self.F[:, i]
                lines.append(f"  f{i + 1}: [{col.min():.6f}, {col.max():.6f}]")

        return "\n".join(lines)

    def summary(self) -> None:
        """Log a summary of the optimization result."""
        for line in self.summary_text().splitlines():
            _logger().info("%s", line)

    @overload
    def front(self, *, return_indices: Literal[False] = False) -> np.ndarray | None: ...

    @overload
    def front(self, *, return_indices: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def front(self, *, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """
        Return non-dominated solutions (first Pareto front).

        Args:
            return_indices: When True, also return indices of the front in F.
        """
        return pareto_filter(self.F, return_indices=return_indices)

    def plot(self, show: bool = True, **kwargs: Any) -> Any:
        """
        Plot the Pareto front (2D or 3D).

        Args:
            show: Whether to display the plot immediately
            **kwargs: Additional arguments passed to scatter plot

        Returns:
            matplotlib Axes object

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If more than 3 objectives
        """
        try:
            import matplotlib

            if not show:
                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

        F_plot = self.front()
        if F_plot is None or len(F_plot) == 0:
            raise ValueError("No solutions to plot")

        n_obj = self.n_objectives
        if n_obj == 2:
            fig, ax = plt.subplots()
            ax.scatter(F_plot[:, 0], F_plot[:, 1], **kwargs)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_title("Pareto Front")
        elif n_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], **kwargs)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            ax.set_title("Pareto Front")
        else:
            raise ValueError(f"Cannot plot {n_obj} objectives. Use to_dataframe() for analysis.")

        if show:
            plt.show()
        return ax

    def best(self, method: str = "knee") -> dict[str, Any]:
        """
        Select a single 'best' solution from the Pareto front.

        Args:
            method: Selection method - 'knee' (default), 'min_f1', 'min_f2', 'balanced'

        Returns:
            Dictionary with 'X' (decision vars), 'F' (objectives),
            'index' (position in original F/X), and 'front_index' (position in the front)
        """
        if self.F is None or len(self.F) == 0:
            raise ValueError("No solutions available")

        front = self.front(return_indices=True)
        if front is None:
            raise ValueError("No solutions available")
        front_F, front_idx = front
        if len(front_F) == 0:
            raise ValueError("No solutions available")

        if method == "knee":
            # Simple knee point: minimize normalized L1 distance
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.sum(axis=1)))
        elif method == "min_f1":
            front_pos = int(np.argmin(front_F[:, 0]))
        elif method == "min_f2":
            front_pos = int(np.argmin(front_F[:, 1]))
        elif method == "balanced":
            # Minimize max normalized objective
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise ValueError(f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced")

        idx = int(front_idx[front_pos])
        X_sel = None
        if self.X is not None:
            try:
                X_sel = self.X[idx]
            except Exception:
                X_sel = None
        return {
            "X": X_sel,
            "F": self.F[idx],
            "index": idx,
            "front_index": front_pos,
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

        if self.F is None or len(self.F) == 0:
            return pd.DataFrame()

        n_obj = self.n_objectives
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
        import json
        from pathlib import Path

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.F is not None:
            np.savetxt(out_dir / "FUN.csv", self.F, delimiter=",")
        if self.X is not None:
            np.savetxt(out_dir / "X.csv", self.X, delimiter=",")

        metadata = {
            "n_solutions": len(self),
            "n_objectives": self.n_objectives,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        _logger().info("Results saved to %s", out_dir)

    def explore(self, title: str = "Pareto Front Explorer") -> Any:
        """
        Launch interactive Plotly dashboard for exploring the Pareto front.

        Opens a browser window with an interactive scatter plot where you can:
        - Hover over points to see objective values
        - Click and drag to zoom
        - Box select to focus on regions
        - Download as PNG

        Args:
            title: Title for the plot

        Returns:
            Plotly Figure object

        Raises:
            ImportError: If plotly is not installed
            ValueError: If no solutions or >3 objectives
        """
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for explore(). Install with: pip install plotly") from exc

        F_plot = self.front()
        if F_plot is None or len(F_plot) == 0:
            raise ValueError("No solutions to explore")

        n_obj = self.n_objectives
        n_solutions = len(F_plot)

        # Build hover text with decision variables
        hover_texts = []
        for i in range(n_solutions):
            lines = [f"<b>Solution {i}</b>"]
            for j in range(n_obj):
                lines.append(f"f{j + 1}: {F_plot[i, j]:.4f}")
            if self.X is not None and i < len(self.X):
                lines.append("<br><b>Decision Variables</b>")
                n_show = min(5, self.X.shape[1])
                for k in range(n_show):
                    lines.append(f"x{k + 1}: {self.X[i, k]:.4f}")
                if self.X.shape[1] > 5:
                    lines.append(f"... ({self.X.shape[1] - 5} more)")
            hover_texts.append("<br>".join(lines))

        if n_obj == 2:
            fig = go.Figure(
                data=go.Scatter(
                    x=F_plot[:, 0],
                    y=F_plot[:, 1],
                    mode="markers",
                    marker=dict(size=10, color="royalblue", opacity=0.7),
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            )
            fig.update_layout(title=title, xaxis_title="f1", yaxis_title="f2", hovermode="closest")
        elif n_obj == 3:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=F_plot[:, 0],
                    y=F_plot[:, 1],
                    z=F_plot[:, 2],
                    mode="markers",
                    marker=dict(size=6, color="royalblue", opacity=0.7),
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            )
            fig.update_layout(title=title, scene=dict(xaxis_title="f1", yaxis_title="f2", zaxis_title="f3"))
        else:
            raise ValueError(f"Cannot visualize {n_obj} objectives interactively. Use to_dataframe() for analysis.")

        fig.show()
        return fig

    def to_latex(
        self,
        caption: str = "Optimization Results",
        label: str = "tab:results",
        precision: int = 4,
    ) -> str:
        """
        Generate a LaTeX table from the optimization results.

        Args:
            caption: Table caption
            label: LaTeX label for referencing
            precision: Decimal places for numbers

        Returns:
            LaTeX table code as string

        Example:
            >>> result = vamos.optimize("zdt1")
            >>> print(result.to_latex())
        """
        if self.F is None or len(self.F) == 0:
            return "% No solutions to display"

        n_obj = self.n_objectives
        n_sol = len(self)

        # Build header
        obj_headers = " & ".join([f"$f_{{{i + 1}}}$" for i in range(n_obj)])
        header = f"Solution & {obj_headers}"
        col_spec = "l" + "r" * n_obj

        # Build rows - show summary stats
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            f"{header} \\\\",
            r"\midrule",
        ]

        # Summary statistics
        f_min = self.F.min(axis=0)
        f_max = self.F.max(axis=0)
        f_mean = self.F.mean(axis=0)
        f_std = self.F.std(axis=0)

        min_vals = " & ".join([f"{v:.{precision}f}" for v in f_min])
        max_vals = " & ".join([f"{v:.{precision}f}" for v in f_max])
        mean_vals = " & ".join([f"{v:.{precision}f}" for v in f_mean])
        std_vals = " & ".join([f"{v:.{precision}f}" for v in f_std])

        lines.append(f"Min & {min_vals} \\\\")
        lines.append(f"Max & {max_vals} \\\\")
        lines.append(f"Mean & {mean_vals} \\\\")
        lines.append(f"Std & {std_vals} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                "\\\\[0.5em]",
                f"\\footnotesize{{$n = {n_sol}$ solutions}}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)


__all__ = ["OptimizationResult"]

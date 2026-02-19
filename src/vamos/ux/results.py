"""
Presentation and export helpers for OptimizationResult-like objects.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.metrics.pareto import pareto_filter


class ResultLike(Protocol):
    F: NDArray[Any] | None
    X: NDArray[Any] | None
    meta: Mapping[str, Any]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def result_summary_text(result: ResultLike) -> str:
    """Return a human-readable summary string for a result object."""
    meta = result.meta
    algo = meta.get("algorithm")
    seed = meta.get("seed")
    engine = meta.get("engine")
    F = result.F

    n_solutions = int(F.shape[0]) if F is not None and len(F) > 0 else 0
    n_objectives = int(F.shape[1]) if F is not None and len(F) > 0 else 0

    lines = [
        "=== Optimization Result ===",
        *([f"Algorithm: {algo}"] if algo else []),
        *([f"Engine: {engine}"] if engine else []),
        *([f"Seed: {seed}"] if seed is not None else []),
        f"Solutions: {n_solutions}",
        f"Objectives: {n_objectives}",
    ]

    if F is not None and len(F) > 0:
        lines.append("Objective ranges:")
        for i in range(n_objectives):
            col = F[:, i]
            lines.append(f"  f{i + 1}: [{col.min():.6f}, {col.max():.6f}]")

    return "\n".join(lines)


def log_result_summary(result: ResultLike, *, logger: logging.Logger | None = None) -> None:
    """Log a summary of the optimization result."""
    active_logger = logger or _logger()
    for line in result_summary_text(result).splitlines():
        active_logger.info("%s", line)


def plot_result_front(
    result: ResultLike,
    show: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Plot the Pareto front (2D or 3D) for a result object.

    Args:
        show: Whether to display the plot immediately.
        title: Optional plot title. Defaults to "Pareto Front".
        **kwargs: Additional arguments passed to scatter plot.
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    F_plot = pareto_filter(result.F, return_indices=False)
    if F_plot is None or len(F_plot) == 0:
        raise ValueError("No solutions to plot")

    F_plot = np.asarray(F_plot, dtype=float)
    n_obj = int(F_plot.shape[1]) if F_plot.ndim == 2 else 0
    plot_title = "Pareto Front" if title is None else title
    ax3d: Any | None = None
    if n_obj == 2:
        fig, ax = plt.subplots()
        ax.scatter(F_plot[:, 0], F_plot[:, 1], **kwargs)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_title(plot_title)
    elif n_obj == 3:
        fig = plt.figure()
        ax3d = cast(Any, fig.add_subplot(111, projection="3d"))
        ax = ax3d
        ax3d.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], **kwargs)
        ax3d.set_xlabel("f1")
        ax3d.set_ylabel("f2")
        ax3d.set_zlabel("f3")
        ax3d.set_title(plot_title)
    else:
        raise ValueError(f"Cannot plot {n_obj} objectives. Use result_to_dataframe() for analysis.")

    if show:
        plt.show()
    return ax


def result_to_dataframe(result: ResultLike) -> Any:
    """
    Convert results to a pandas DataFrame.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for result_to_dataframe(). Install with: pip install pandas") from exc

    F = result.F
    if F is None or len(F) == 0:
        return pd.DataFrame()

    n_obj = int(F.shape[1])
    data = {f"f{i + 1}": F[:, i] for i in range(n_obj)}

    X = result.X
    if X is not None:
        n_var = int(X.shape[1])
        for i in range(n_var):
            data[f"x{i + 1}"] = X[:, i]

    return pd.DataFrame(data)


def save_result(result: ResultLike, path: str) -> None:
    """
    Save results to a directory (CSV files for F, X, and metadata).
    """
    import json
    from pathlib import Path

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    F = result.F
    X = result.X
    if F is not None:
        np.savetxt(out_dir / "FUN.csv", F, delimiter=",")
    if X is not None:
        np.savetxt(out_dir / "X.csv", X, delimiter=",")

    n_solutions = int(F.shape[0]) if F is not None and len(F) > 0 else 0
    n_objectives = int(F.shape[1]) if F is not None and len(F) > 0 else 0
    metadata = {
        "n_solutions": n_solutions,
        "n_objectives": n_objectives,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    _logger().info("Results saved to %s", out_dir)


def explore_result_front(result: ResultLike, title: str = "Pareto Front Explorer") -> Any:
    """
    Launch an interactive Plotly dashboard for exploring the Pareto front.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for explore_result_front(). Install with: pip install plotly") from exc

    F_plot = pareto_filter(result.F, return_indices=False)
    if F_plot is None or len(F_plot) == 0:
        raise ValueError("No solutions to explore")

    F_plot = np.asarray(F_plot, dtype=float)
    n_obj = int(F_plot.shape[1]) if F_plot.ndim == 2 else 0
    n_solutions = len(F_plot)
    X = result.X

    hover_texts = []
    for i in range(n_solutions):
        lines = [f"<b>Solution {i}</b>"]
        for j in range(n_obj):
            lines.append(f"f{j + 1}: {F_plot[i, j]:.4f}")
        if X is not None and i < len(X):
            lines.append("<br><b>Decision Variables</b>")
            n_show = min(5, X.shape[1])
            for k in range(n_show):
                lines.append(f"x{k + 1}: {X[i, k]:.4f}")
            if X.shape[1] > 5:
                lines.append(f"... ({X.shape[1] - 5} more)")
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
        raise ValueError(f"Cannot visualize {n_obj} objectives interactively. Use result_to_dataframe() for analysis.")

    fig.show()
    return fig


def result_to_latex(
    result: ResultLike,
    caption: str = "Optimization Results",
    label: str = "tab:results",
    precision: int = 4,
) -> str:
    """
    Generate a LaTeX table from the optimization results.
    """
    F = result.F
    if F is None or len(F) == 0:
        return "% No solutions to display"

    n_obj = int(F.shape[1])
    n_sol = int(F.shape[0])

    obj_headers = " & ".join([f"$f_{{{i + 1}}}$" for i in range(n_obj)])
    header = f"Solution & {obj_headers}"
    col_spec = "l" + "r" * n_obj

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

    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    f_mean = F.mean(axis=0)
    f_std = F.std(axis=0)

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


__all__ = [
    "ResultLike",
    "result_summary_text",
    "log_result_summary",
    "plot_result_front",
    "explore_result_front",
    "result_to_dataframe",
    "result_to_latex",
    "save_result",
]

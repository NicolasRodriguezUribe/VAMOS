from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from vamos.foundation.metrics.pareto import pareto_filter


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.array([], dtype=bool)
    n_points = points.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        dominates = np.all(points[i] <= points, axis=1) & np.any(points[i] < points, axis=1)
        dominates[i] = False
        mask[dominates] = False
    return mask


def _problem_output_dir(selection: Any, output_root: str) -> str:
    safe = selection.spec.label.replace(" ", "_").upper()
    return os.path.join(output_root, f"{safe}")


def plot_pareto_front(
    results: Iterable[dict[str, Any]],
    selection: Any,
    *,
    output_root: str,
    title: str,
) -> str | None:
    plot_entries = []
    for res in results:
        F = res.get("F")
        if F is None or F.size == 0:
            continue
        algo = res.get("algorithm", "unknown").upper()
        engine = res.get("engine")
        label = f"{algo} ({engine})" if engine else algo
        filtered = pareto_filter(F)
        if filtered is None or filtered.size == 0:
            continue
        plot_entries.append((label, np.asarray(filtered, dtype=float)))
    if not plot_entries:
        return None
    n_obj = plot_entries[0][1].shape[1]
    if n_obj < 2:
        _logger().warning("Pareto visualization requires at least two objectives; skipping plot.")
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        _logger().warning(
            "matplotlib is required for plotting the Pareto front (skipping plot: %s).",
            exc,
        )
        return None

    dims = 3 if n_obj >= 3 else 2
    if dims == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("Objective 3")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")

    cmap = plt.cm.get_cmap("tab10", len(plot_entries))
    for idx, (label, values) in enumerate(plot_entries):
        coords = values[:, :dims]
        color = cmap(idx)
        if dims == 3:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=label, s=22, alpha=0.7, color=color)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], label=label, s=35, alpha=0.8, color=color)

    all_points_full = np.vstack([entry[1] for entry in plot_entries])
    front_mask = _non_dominated_mask(all_points_full)
    if np.any(front_mask):
        projected = all_points_full[front_mask][:, :dims]
        if dims == 2:
            order = np.argsort(projected[:, 0])
            projected = projected[order]
            ax.plot(
                projected[:, 0],
                projected[:, 1],
                color="black",
                linewidth=2,
                label="Pareto front (union)",
            )
        else:
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                projected[:, 2],
                color="black",
                s=40,
                label="Pareto front (union)",
                marker="x",
            )

    plot_title = f"Pareto front - {selection.spec.label}"
    if n_obj > dims:
        plot_title += f" (showing first {dims} objectives)"
    ax.set_title(plot_title)
    ax.legend()
    fig.tight_layout()

    os.makedirs(_problem_output_dir(selection, output_root), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pareto_front_{timestamp}.png"
    plot_path = os.path.join(_problem_output_dir(selection, output_root), filename)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    _logger().info("Pareto front plot saved to: %s", plot_path)
    return plot_path

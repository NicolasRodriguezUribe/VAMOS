from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _get_ax(ax: plt.Axes | None, projection: str | None = None) -> plt.Axes:
    if ax is not None:
        return ax
    fig = plt.figure()
    if projection is None:
        return fig.add_subplot(111)
    return fig.add_subplot(111, projection=projection)


def plot_pareto_front_2d(
    F: np.ndarray,
    ax: plt.Axes | None = None,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    show: bool = False,
) -> plt.Axes:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] != 2:
        raise ValueError("F must have shape (n_points, 2) for 2D Pareto plot.")
    ax = _get_ax(ax)
    ax.scatter(F[:, 0], F[:, 1], s=20, alpha=0.7)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    if show:
        plt.show()
    return ax


def plot_pareto_front_3d(
    F: np.ndarray,
    ax: plt.Axes | None = None,
    labels: tuple[str, str, str] | None = None,
    title: str | None = None,
    show: bool = False,
) -> plt.Axes:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("F must have shape (n_points, 3) for 3D Pareto plot.")
    ax = _get_ax(ax, projection="3d")
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=20, alpha=0.7)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    if title:
        ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    if show:
        plt.show()
    return ax


def plot_parallel_coordinates(
    F: np.ndarray,
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show: bool = False,
) -> plt.Axes:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2-dimensional for parallel coordinates.")
    n_obj = F.shape[1]
    data = F.copy()
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    span = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    data = (data - min_vals) / span
    ax = _get_ax(ax)
    for row in data:
        ax.plot(range(n_obj), row, alpha=0.5)
    ax.set_xlim(0, n_obj - 1)
    if labels and len(labels) == n_obj:
        ax.set_xticks(range(n_obj))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_hv_convergence(
    evals: np.ndarray,
    hv_values: np.ndarray,
    ax: plt.Axes | None = None,
    label: str | None = None,
    title: str | None = "Hypervolume convergence",
    show: bool = False,
) -> plt.Axes:
    evals = np.asarray(evals, dtype=float)
    hv_values = np.asarray(hv_values, dtype=float)
    if evals.shape != hv_values.shape:
        raise ValueError("evals and hv_values must have the same shape.")
    ax = _get_ax(ax)
    ax.plot(evals, hv_values, label=label)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    if show:
        plt.show()
    return ax

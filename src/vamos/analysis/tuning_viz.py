"""
Notebook-friendly helpers for inspecting tuning results and fronts.
Uses only NumPy/Pandas/Matplotlib.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vamos.analysis.objective_reduction import ObjectiveReducer


def tuning_result_to_dataframe(tuning_result: Any, param_names: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Convert a TuningResult-like object to a DataFrame.

    Expected keys/attributes: unit_vectors (X), objectives (F), assignments/configs (optional).
    """
    X = getattr(tuning_result, "unit_vectors", None) or getattr(tuning_result, "X_nd", None) or tuning_result[0]
    F = getattr(tuning_result, "objectives", None) or getattr(tuning_result, "F_nd", None) or tuning_result[1]
    assignments = getattr(tuning_result, "assignments", None) or getattr(tuning_result, "configs", None) or []
    X = np.asarray(X)
    F = np.asarray(F)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(X.shape[1])]
    rows = []
    for i in range(X.shape[0]):
        row = {param_names[j]: X[i, j] for j in range(X.shape[1])}
        for k, val in enumerate(F[i]):
            row[f"obj_{k}"] = val
        if assignments:
            row["assignment"] = assignments[i]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tuning_scatter(df: pd.DataFrame, x_param: str, y_param: str, color_by: str = "obj_0"):
    """Scatter plot of hyperparameters colored by an objective/metric."""
    plt.figure()
    sc = plt.scatter(df[x_param], df[y_param], c=df[color_by], cmap="viridis", edgecolor="k", alpha=0.8)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    cbar = plt.colorbar(sc)
    cbar.set_label(color_by)
    plt.tight_layout()
    return plt.gca()


def plot_objective_tradeoff(df: pd.DataFrame, obj_x: str = "obj_0", obj_y: str = "obj_1"):
    """Plot objective trade-off scatter."""
    plt.figure()
    plt.scatter(df[obj_x], df[obj_y], color="tab:blue", edgecolor="k", alpha=0.8)
    plt.xlabel(obj_x)
    plt.ylabel(obj_y)
    plt.tight_layout()
    return plt.gca()


def plot_reduced_front(F: np.ndarray, labels: Sequence[str] | None = None, target_dim: int = 2, method: str = "angle"):
    """
    Reduce a front to 2D/3D and plot it.
    """
    reducer = ObjectiveReducer(method=method)
    selected = reducer.reduce(F, target_dim=target_dim)
    F_red = F[:, selected]
    labels = labels or [f"f{i}" for i in range(F_red.shape[1])]
    if F_red.shape[1] == 2:
        plt.figure()
        plt.scatter(F_red[:, 0], F_red[:, 1], color="tab:orange", edgecolor="k", alpha=0.8)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.tight_layout()
        return plt.gca()
    if F_red.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F_red[:, 0], F_red[:, 1], F_red[:, 2], color="tab:orange", alpha=0.8)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        fig.tight_layout()
        return ax
    raise ValueError("Reduced front must have at least 2 objectives.")


def study_results_to_dataframe(study_results: Iterable[Any]) -> pd.DataFrame:
    """
    Convert StudyRunner results to a tidy DataFrame.
    Expects each entry with .selection.spec.key, .metrics dict fields.
    """
    rows = []
    for res in study_results:
        metrics = res.metrics
        rows.append(
            {
                "problem": res.selection.spec.key,
                "algorithm": metrics.get("algorithm"),
                "engine": metrics.get("engine"),
                "seed": metrics.get("seed", None),
                "hv": metrics.get("hv"),
                "time_ms": metrics.get("time_ms"),
                "evals": metrics.get("evaluations"),
                "spread": metrics.get("spread"),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_algorithm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean/std summary grouped by problem and algorithm.
    """
    grouped = df.groupby(["problem", "algorithm"]).agg(["mean", "std"])
    grouped.columns = ["_".join(col).rstrip("_") for col in grouped.columns]
    return grouped.reset_index()

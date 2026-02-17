"""
Hypervolume utilities for orchestration layers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.core.experiment_config import HV_REFERENCE_OFFSET


def build_hv_stop_config(
    hv_threshold: float | None,
    hv_reference_front: str | None,
    problem_key: str,
    n_obj: int | None = None,
) -> dict[str, object] | None:
    """
    Build an early-stop configuration for hypervolume-based termination.
    """
    if hv_threshold is None:
        return None
    front_path = resolve_reference_front_path(problem_key, hv_reference_front, n_obj=n_obj)
    if front_path is None:
        raise ValueError(f"No reference front found for problem '{problem_key}'. Provide --hv-reference-front or add a built-in front.")
    reference_front = np.loadtxt(front_path, delimiter=",")
    if reference_front.ndim != 2 or reference_front.shape[1] < 2:
        raise ValueError(f"Reference front '{front_path}' must be a 2D array with at least two objectives.")
    max_vals = reference_front.max(axis=0)
    margin = np.maximum(0.1 * np.maximum(np.abs(max_vals), 1.0), 5.0)
    if problem_key.lower() == "zdt6":
        margin = np.maximum(margin, 10.0)
    ref_point = max_vals + margin
    hv_full = hypervolume(reference_front, ref_point)
    if hv_full <= 0.0:
        raise ValueError(f"Reference front '{front_path}' produced a non-positive hypervolume; check the data.")
    threshold_fraction = float(hv_threshold)
    return {
        "target_value": hv_full * threshold_fraction,
        "threshold_fraction": threshold_fraction,
        "reference_point": ref_point.astype(float, copy=False).tolist(),
        "reference_front_path": str(front_path),
    }


def compute_hv_reference(fronts: Iterable[np.ndarray]) -> np.ndarray:
    """
    Build a hypervolume reference point that weakly dominates all supplied fronts.
    A small margin (HV_REFERENCE_OFFSET) is added to keep the reference outside
    the sampled region even when solutions lie on the boundary.
    """
    collected = []
    for idx, front in enumerate(fronts):
        if front is None:
            continue
        arr = np.asarray(front, dtype=float)
        if arr.size == 0:
            continue
        if arr.ndim != 2:
            raise ValueError(f"Front {idx} must be a 2D array; got shape {arr.shape}.")
        collected.append(arr)

    if not collected:
        raise ValueError("At least one non-empty front is required to compute a reference point.")

    n_obj = collected[0].shape[1]
    for arr in collected:
        if arr.shape[1] != n_obj:
            raise ValueError("All fronts must have the same number of objectives.")

    stacked = np.vstack(collected)
    max_vals = stacked.max(axis=0)
    margin = np.maximum(np.abs(max_vals) * HV_REFERENCE_OFFSET, HV_REFERENCE_OFFSET)
    return np.asarray(max_vals + margin, dtype=float)


__all__ = ["build_hv_stop_config", "compute_hv_reference"]

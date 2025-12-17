from __future__ import annotations

import numpy as np
from typing import Sequence


def _is_finite_array(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()


def compute_hypervolume(F: np.ndarray, ref_point: Sequence[float]) -> float:
    """Compute (exact) hypervolume for 2D minimization fronts.

    Parameters
    - F: array-like shape (n_points, 2) of objective values (minimization)
    - ref_point: sequence of length 2 with reference point (worst values)

    Returns
    - hypervolume (float)

    Notes
    - This implementation supports 2-objective minimization problems only.
    - For higher-dimensional hypervolume, use a dedicated backend (e.g., Moocore)
      via the `vamos.foundation.metrics.moocore_indicators` helpers.
    """
    F = np.asarray(F)
    ref = np.asarray(ref_point, dtype=float)

    if F.ndim != 2 or F.shape[1] != 2:
        raise ValueError("compute_hypervolume currently supports 2D fronts only")

    if not _is_finite_array(F) or not np.isfinite(ref).all():
        raise ValueError("F and ref_point must contain finite numbers")

    # Filter dominated/worse points relative to reference
    # Keep only points that are strictly better than ref in at least one objective
    mask = np.any(F < ref, axis=1)
    pts = F[mask]

    if pts.size == 0:
        return 0.0

    # Remove points that are dominated by others (keep Pareto front)
    # For 2D minimization: sort by f1 ascending, then keep those with strictly decreasing f2
    idx = np.argsort(pts[:, 0])
    sorted_pts = pts[idx]

    # Sweep from left (small f1) to right, keeping non-dominated in f2
    pareto = []
    best_f2 = np.inf
    for x, y in sorted_pts:
        if y < best_f2:
            pareto.append((x, y))
            best_f2 = y

    pareto = np.array(pareto)

    # Compute hypervolume as sum of rectangles between successive Pareto points
    hv = 0.0
    prev_f1 = ref[0]
    # iterate pareto points in reverse (from worst f1 to best) to form rectangles
    for x, y in pareto[::-1]:
        width = prev_f1 - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = x

    return float(max(hv, 0.0))

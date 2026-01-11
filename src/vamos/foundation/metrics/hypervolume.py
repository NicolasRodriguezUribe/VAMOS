from __future__ import annotations

import numpy as np
from typing import Sequence


def _is_finite_array(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


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

    # Filter points inside the reference box (dominated by ref in ALL objectives)
    mask = np.all(F <= ref, axis=1)
    pts = F[mask]

    if pts.size == 0:
        return 0.0

    # Remove points that are dominated by others (keep Pareto front)
    # For 2D minimization: sort by f1 ascending, then keep those with strictly decreasing f2
    idx = np.argsort(pts[:, 0])
    sorted_pts = pts[idx]

    # Sweep from left (small f1) to right, keeping non-dominated in f2
    pareto: list[tuple[float, float]] = []
    best_f2 = np.inf
    for x, y in sorted_pts:
        if y < best_f2:
            pareto.append((x, y))
            best_f2 = y

    pareto_arr = np.asarray(pareto, dtype=float)

    # Compute hypervolume as sum of rectangles between successive Pareto points
    hv = 0.0
    prev_f1 = ref[0]
    # iterate pareto points in reverse (from worst f1 to best) to form rectangles
    for x, y in pareto_arr[::-1]:
        width = prev_f1 - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = x

    return float(max(hv, 0.0))


# General hypervolume utilities (moocore/libhv fallback).
_MOOCORE = None
_LIBHV_MODULE = None
_LIBHV_HV_FN = None
_LIBHV_CLASS = None
_OPTIONAL_LOADED = False


def _load_optional_backends() -> None:
    global _MOOCORE, _LIBHV_MODULE, _LIBHV_HV_FN, _LIBHV_CLASS, _OPTIONAL_LOADED
    if _OPTIONAL_LOADED:
        return
    _OPTIONAL_LOADED = True
    try:  # Prefer the MooCore C backend when available.
        import moocore as moocore_module
    except ImportError:  # pragma: no cover - optional dependency
        moocore_module = None
    _MOOCORE = moocore_module

    try:  # pragma: no cover - optional dependency
        import libhv as libhv_module
    except ImportError:
        libhv_module = None

    if libhv_module is None:
        _LIBHV_MODULE = None
        _LIBHV_HV_FN = None
        _LIBHV_CLASS = None
        return

    hv_fn = getattr(libhv_module, "hypervolume", None)
    hv_cls = getattr(libhv_module, "HyperVolume", None)
    if hv_fn is None and hv_cls is None:
        _LIBHV_MODULE = None
        _LIBHV_HV_FN = None
        _LIBHV_CLASS = None
        return

    _LIBHV_MODULE = libhv_module
    _LIBHV_HV_FN = hv_fn
    _LIBHV_CLASS = hv_cls


def hypervolume(points: np.ndarray, reference_point: np.ndarray, *, allow_ref_expand: bool = True) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if points.shape[0] == 0:
        return 0.0

    ref = _validate_reference_point(points, reference_point, allow_ref_expand=allow_ref_expand)
    _load_optional_backends()
    if _MOOCORE is not None:
        data = np.ascontiguousarray(points, dtype=np.float64)
        return float(_MOOCORE.hypervolume(data, ref))
    if _LIBHV_MODULE is not None:
        return float(_hypervolume_with_libhv(points, ref))
    return _hypervolume_impl(points, ref)


def hypervolume_contributions(
    points: np.ndarray,
    reference_point: np.ndarray,
    *,
    allow_ref_expand: bool = True,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=float)

    ref = _validate_reference_point(points, reference_point, allow_ref_expand=allow_ref_expand)
    _load_optional_backends()
    if _MOOCORE is not None:
        data = np.ascontiguousarray(points, dtype=np.float64)
        return np.asarray(_MOOCORE.hv_contributions(data, ref), dtype=float)
    if points.shape[1] == 2:
        return _hypervolume_contributions_2d(points, ref)

    return _hypervolume_contributions_generic(points, ref)


def _hypervolume_with_libhv(points: np.ndarray, ref: np.ndarray) -> float:
    data = np.ascontiguousarray(points, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if _LIBHV_HV_FN is not None:
        return float(_LIBHV_HV_FN(data, ref_arr))
    hv_class = _LIBHV_CLASS
    if hv_class is None:  # pragma: no cover - guarded by initialization
        raise RuntimeError("libhv is imported without HyperVolume support.")
    hv_obj = hv_class(ref_arr)
    compute = getattr(hv_obj, "compute", None) or getattr(hv_obj, "compute_from_points", None)
    if compute is None:
        raise AttributeError("libhv.HyperVolume object lacks a compute method.")
    return float(compute(data))


def _hypervolume_impl(points: np.ndarray, reference_point: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    ref = np.asarray(reference_point, dtype=float)
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return float(np.max(widths))
    if n_obj == 2:
        return _hypervolume_2d(points, ref)
    if n_obj == 3:
        return _hypervolume_3d(points, ref)
    return _hypervolume_recursive(points, ref)


def _validate_reference_point(
    points: np.ndarray,
    reference_point: np.ndarray,
    *,
    allow_ref_expand: bool = True,
) -> np.ndarray:
    ref = np.asarray(reference_point, dtype=float)
    if ref.ndim != 1:
        raise ValueError("reference_point must be a 1D array.")
    if ref.shape[0] != points.shape[1]:
        raise ValueError("reference_point dimensionality mismatch.")
    if np.any(points > ref):
        if not allow_ref_expand:
            raise ValueError("reference_point must dominate all points.")
        # Expand reference point to dominate all points to avoid runtime errors.
        ref = np.maximum(ref, points.max(axis=0) + 1e-9)
    return ref


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    order = np.argsort(points[:, 0], kind="mergesort")
    sorted_points = points[order]
    widths = np.maximum(ref[0] - sorted_points[:, 0], 0.0)
    prev_f2 = np.minimum.accumulate(np.concatenate(([ref[1]], sorted_points[:-1, 1])))
    heights = np.maximum(prev_f2 - sorted_points[:, 1], 0.0)
    return float(np.sum(widths * heights))


def _hypervolume_3d(points: np.ndarray, ref: np.ndarray) -> float:
    order = np.argsort(points[:, 2])
    sorted_points = points[order]
    hv = 0.0
    prev_f3 = ref[2]
    for end in range(sorted_points.shape[0] - 1, -1, -1):
        f3 = sorted_points[end, 2]
        height = max(prev_f3 - f3, 0.0)
        if height <= 0.0:
            continue
        slice_points = sorted_points[: end + 1, :2]
        slab = _hypervolume_2d(slice_points, ref[:2])
        hv += slab * height
        prev_f3 = f3
    return hv


def _hypervolume_recursive(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Recursive slicing algorithm that works for >= 4 objectives.
    """
    if points.size == 0:
        return 0.0
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return np.max(widths) if widths.size else 0.0

    order = np.argsort(points[:, n_obj - 1])
    sorted_points = points[order]

    hv = 0.0
    bound = ref[n_obj - 1]
    while sorted_points.shape[0] > 0:
        current = sorted_points[-1, n_obj - 1]
        height = bound - current
        if height > 0.0:
            reduced = sorted_points[:, : n_obj - 1]
            hv += _hypervolume_recursive(reduced, ref[: n_obj - 1]) * height
            bound = current
        sorted_points = sorted_points[:-1]
    return hv


def _hypervolume_contributions_2d(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)

    unique_points, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    unique_contribs = np.zeros(unique_points.shape[0], dtype=float)

    if unique_points.shape[0]:
        order = np.lexsort((unique_points[:, 1], unique_points[:, 0]))
        sorted_points = unique_points[order]

        prev_min = np.concatenate(([np.inf], np.minimum.accumulate(sorted_points[:-1, 1])))
        nd_sorted = sorted_points[:, 1] < prev_min
        nd_order = order[nd_sorted]
        if nd_order.size:
            nd_points = unique_points[nd_order]
            x = nd_points[:, 0]
            y = nd_points[:, 1]
            x_next = np.concatenate((x[1:], [ref[0]]))
            y_prev = np.concatenate(([ref[1]], y[:-1]))
            nd_contrib = np.maximum(x_next - x, 0.0) * np.maximum(y_prev - y, 0.0)
            unique_contribs[nd_order] = nd_contrib

    contributions = unique_contribs[inverse]
    contributions[counts[inverse] > 1] = 0.0
    return contributions


def _hypervolume_contributions_generic(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    contributions = np.empty(points.shape[0], dtype=float)
    hv_full = _hypervolume_impl(points, ref)
    for i in range(points.shape[0]):
        without_i = np.delete(points, i, axis=0)
        hv_without = _hypervolume_impl(without_i, ref)
        contributions[i] = hv_full - hv_without
    return contributions

"""
Hypervolume utilities that prioritize fast native implementations (moocore, libhv)
while keeping the previous NumPy/Numba routines as a fallback. All functions assume
minimization problems where the reference point dominates the solutions.
"""
from __future__ import annotations

import numpy as np

try:  # Prefer the MooCore C backend when available.
    import moocore as _moocore  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

_LIBHV_MODULE = None
_LIBHV_HV_FN = None
_LIBHV_CLASS = None
try:  # pragma: no cover - optional dependency
    import libhv as _libhv_module  # type: ignore
except ImportError:
    _libhv_module = None
else:
    _LIBHV_HV_FN = getattr(_libhv_module, "hypervolume", None)
    _LIBHV_CLASS = getattr(_libhv_module, "HyperVolume", None)
    if _LIBHV_HV_FN is None and _LIBHV_CLASS is None:
        _libhv_module = None
    else:
        _LIBHV_MODULE = _libhv_module


def hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if points.shape[0] == 0:
        return 0.0

    ref = _validate_reference_point(points, reference_point)
    if _moocore is not None:
        data = np.ascontiguousarray(points, dtype=np.float64)
        return float(_moocore.hypervolume(data, ref))
    if _LIBHV_MODULE is not None:
        return float(_hypervolume_with_libhv(points, ref))
    return _hypervolume_impl(points, ref)


def hypervolume_contributions(points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=float)

    ref = _validate_reference_point(points, reference_point)
    if _moocore is not None:
        data = np.ascontiguousarray(points, dtype=np.float64)
        return np.asarray(_moocore.hv_contributions(data, ref), dtype=float)
    if points.shape[1] == 2:
        return _hypervolume_contributions_2d(points, ref)

    return _hypervolume_contributions_generic(points, ref)


def _hypervolume_with_libhv(points: np.ndarray, ref: np.ndarray) -> float:
    data = np.ascontiguousarray(points, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    if _LIBHV_HV_FN is not None:
        return _LIBHV_HV_FN(data, ref_arr)
    hv_class = _LIBHV_CLASS
    if hv_class is None:  # pragma: no cover - guarded by initialization
        raise RuntimeError("libhv is imported without HyperVolume support.")
    hv_obj = hv_class(ref_arr)
    compute = getattr(hv_obj, "compute", None) or getattr(hv_obj, "compute_from_points", None)
    if compute is None:
        raise AttributeError("libhv.HyperVolume object lacks a compute method.")
    return compute(data)


def _hypervolume_impl(points: np.ndarray, reference_point: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    ref = _validate_reference_point(points, reference_point)
    n_obj = points.shape[1]
    if n_obj == 1:
        widths = np.maximum(ref[0] - points[:, 0], 0.0)
        return np.max(widths)
    if n_obj == 2:
        return _hypervolume_2d(points, ref)
    if n_obj == 3:
        return _hypervolume_3d(points, ref)
    return _hypervolume_recursive(points, ref)


def _validate_reference_point(points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_point, dtype=float)
    if ref.ndim != 1:
        raise ValueError("reference_point must be a 1D array.")
    if ref.shape[0] != points.shape[1]:
        raise ValueError("reference_point dimensionality mismatch.")
    if np.any(points > ref):
        # Expand reference point to dominate all points to avoid runtime errors.
        ref = np.maximum(ref, points.max(axis=0) + 1e-9)
    return ref


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    order = np.argsort(points[:, 0])
    hv = 0.0
    prev_f2 = ref[1]
    for idx in reversed(order):
        f1, f2 = points[idx]
        width = max(ref[0] - f1, 0.0)
        height = max(prev_f2 - f2, 0.0)
        hv += width * height
        prev_f2 = min(prev_f2, f2)
    return hv


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
    order = np.argsort(points[:, 0])
    contributions = np.zeros(points.shape[0], dtype=float)
    prev_f2 = ref[1]
    for pos in range(order.size - 1, -1, -1):
        idx = order[pos]
        f1, f2 = points[idx]
        width = max(ref[0] - f1, 0.0)
        height = max(prev_f2 - f2, 0.0)
        contributions[idx] = width * height
        prev_f2 = min(prev_f2, f2)
    return contributions


def _hypervolume_contributions_generic(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    contributions = np.empty(points.shape[0], dtype=float)
    hv_full = _hypervolume_impl(points, ref)
    for i in range(points.shape[0]):
        without_i = np.delete(points, i, axis=0)
        hv_without = _hypervolume_impl(without_i, ref)
        contributions[i] = hv_full - hv_without
    return contributions

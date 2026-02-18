from __future__ import annotations

from functools import cache
import os
from pathlib import Path

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.metrics.pareto import pareto_filter

ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_FRONTS_DIR = ROOT_DIR / "src" / "vamos" / "foundation" / "data" / "reference_fronts"
REF_EPS = 1e-6
REF_MODE = os.environ.get("VAMOS_HV_REF_MODE", "pf").strip().lower()

_ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
_DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
_DTLZ_N_OBJ = 3


@cache
def _load_reference_front(problem_name: str) -> np.ndarray:
    name = problem_name.lower()
    path = REFERENCE_FRONTS_DIR / f"{name}.csv"
    if not path.is_file():
        alt = REFERENCE_FRONTS_DIR / f"{problem_name.upper()}.csv"
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"Missing reference front for '{problem_name}': {path}")
    return np.loadtxt(path, delimiter=",")


def _bounds_reference_point(problem_name: str) -> np.ndarray:
    name = problem_name.lower()
    if name in {"zdt1", "zdt2", "zdt3", "zdt6"}:
        return np.array([1.1, 11.0], dtype=float)
    if name == "zdt4":
        n_var = _ZDT_N_VAR[name]
        g_max = 1.0 + 45.0 * (n_var - 1)
        return np.array([1.1, g_max * 1.1], dtype=float)
    if name == "dtlz1":
        k = _DTLZ_N_VAR[name] - _DTLZ_N_OBJ + 1
        g_max = 225.0 * k
        f_max = 0.5 * (1.0 + g_max)
        return np.full(_DTLZ_N_OBJ, f_max * 1.1, dtype=float)
    if name in {"dtlz2", "dtlz4", "dtlz5"}:
        k = _DTLZ_N_VAR[name] - _DTLZ_N_OBJ + 1
        g_max = 0.25 * k
        f_max = 1.0 + g_max
        return np.full(_DTLZ_N_OBJ, f_max * 1.1, dtype=float)
    if name == "dtlz3":
        k = _DTLZ_N_VAR[name] - _DTLZ_N_OBJ + 1
        g_max = 225.0 * k
        f_max = 1.0 + g_max
        return np.full(_DTLZ_N_OBJ, f_max * 1.1, dtype=float)
    if name == "dtlz6":
        k = _DTLZ_N_VAR[name] - _DTLZ_N_OBJ + 1
        g_max = float(k)
        f_max = 1.0 + g_max
        return np.full(_DTLZ_N_OBJ, f_max * 1.1, dtype=float)
    if name == "dtlz7":
        g_max = 10.0
        f_last_max = (1.0 + g_max) * _DTLZ_N_OBJ
        return np.array([1.1, 1.1, f_last_max * 1.1], dtype=float)
    if name.startswith("wfg"):
        return np.array([3.3, 5.5], dtype=float)
    raise ValueError(f"Unsupported problem for bounds reference point: '{problem_name}'")


@cache
def _reference_point(problem_name: str) -> np.ndarray:
    if REF_MODE == "bounds":
        return _bounds_reference_point(problem_name)
    front = _load_reference_front(problem_name)
    if front.ndim != 2 or front.shape[0] == 0:
        raise ValueError(f"Invalid reference front for '{problem_name}': shape={front.shape}")
    return front.max(axis=0) + REF_EPS


@cache
def _reference_hv(problem_name: str) -> float:
    front = _load_reference_front(problem_name)
    ref = _reference_point(problem_name)
    front = front[np.all(front <= ref, axis=1)]
    return hypervolume(front, ref, allow_ref_expand=False) if front.size else 0.0


def compute_hv(F, problem_name: str) -> float:
    """Compute normalized hypervolume using a fixed reference front."""
    if F is None:
        return float("nan")
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.size == 0:
        return 0.0

    front = pareto_filter(F_arr)
    if front is None or front.size == 0:
        return 0.0

    ref = _reference_point(problem_name)
    front = front[np.all(front <= ref, axis=1)]
    if front.size == 0:
        return 0.0

    hv = hypervolume(front, ref, allow_ref_expand=False)
    hv_ref = _reference_hv(problem_name)
    return hv / hv_ref if hv_ref > 0.0 else 0.0


def compute_igd_plus(F, problem_name: str) -> float:
    """Compute IGD+ using a fixed reference front.

    Returns the IGD+ value (lower is better).  Uses moocore if available,
    otherwise falls back to a pure-NumPy implementation.
    """
    if F is None:
        return float("nan")
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.size == 0:
        return float("inf")

    front = pareto_filter(F_arr)
    if front is None or front.size == 0:
        return float("inf")

    ref_front = _load_reference_front(problem_name)

    try:
        import moocore
        return float(moocore.igd_plus(front, ref=ref_front, maximise=False))
    except ImportError:
        pass

    # Pure-NumPy fallback: IGD+ = mean over ref points of min modified distance
    # Modified distance: max(f_i - r_i, 0) for each objective
    diffs = front[np.newaxis, :, :] - ref_front[:, np.newaxis, :]  # (R, N, m)
    diffs = np.maximum(diffs, 0.0)
    dists = np.sqrt((diffs ** 2).sum(axis=2))  # (R, N)
    return float(dists.min(axis=1).mean())


__all__ = [
    "compute_hv",
    "compute_igd_plus",
    "_load_reference_front",
    "_reference_point",
    "_reference_hv",
]

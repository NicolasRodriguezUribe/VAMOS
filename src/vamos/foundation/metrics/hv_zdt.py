from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.components.hypervolume import hypervolume


def get_zdt_reference_front(name: str, n_points: int = 1000) -> np.ndarray:
    """Return an approximate Pareto front for ZDT problems (analytic for 1,2,3,4,6)."""
    name = name.lower()
    t = np.linspace(0.0, 1.0, num=n_points)
    if name == "zdt1":
        f1 = t
        f2 = 1.0 - np.sqrt(t)
    elif name == "zdt2":
        f1 = t
        f2 = 1.0 - np.square(t)
    elif name == "zdt3":
        f1 = t
        g = 1.0 + 9.0 * np.sum(np.zeros_like(t), axis=0)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1) / g
    elif name == "zdt4":
        f1 = t
        f2 = 1.0 - np.sqrt(f1)
    elif name == "zdt6":
        f1 = 1.0 - np.exp(-4.0 * t) * np.power(np.sin(6.0 * np.pi * t), 6)
        f2 = 1.0 - np.square(1.0 - f1)
    else:
        raise ValueError(f"Unsupported ZDT problem '{name}'.")
    return np.vstack((f1, f2)).T


def get_zdt_reference_point(name: str) -> np.ndarray:
    """Return a reasonable reference point for HV of the given ZDT problem."""
    name = name.lower()
    if name in {"zdt1", "zdt2", "zdt3"}:
        return np.array([1.1, 1.1], dtype=float)
    if name == "zdt4":
        return np.array([1.5, 4.0], dtype=float)
    if name == "zdt6":
        return np.array([1.1, 1.1], dtype=float)
    raise ValueError(f"Unsupported ZDT problem '{name}'.")


def compute_normalized_hv(F: np.ndarray, problem_name: str) -> float:
    """
    Compute normalized hypervolume: HV(F)/HV(PF_ref) using ZDT reference fronts.
    """
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] == 0:
        return 0.0
    pf_ref = get_zdt_reference_front(problem_name)
    base_ref = get_zdt_reference_point(problem_name)
    # Ensure the reference point dominates both the population and the reference front.
    max_vals = np.maximum(np.max(pf_ref, axis=0), np.max(F_arr, axis=0))
    ref_point = np.maximum(base_ref, max_vals + 1e-8)
    hv_pf = hypervolume(pf_ref, ref_point)
    hv_pop = hypervolume(F_arr, ref_point)
    if hv_pf <= 0.0:
        return 0.0
    return hv_pop / hv_pf


__all__ = ["get_zdt_reference_front", "get_zdt_reference_point", "compute_normalized_hv"]

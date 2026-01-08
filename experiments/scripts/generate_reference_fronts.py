from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pymoo.problems import get_problem

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_OUT_DIR = ROOT_DIR / "data" / "reference_fronts"
# Only ZDT reference fronts are packaged with the library for CLI defaults.
PACKAGE_OUT_DIR = ROOT_DIR / "src" / "vamos" / "foundation" / "data" / "reference_fronts"

ZDT_PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DTLZ_PROBLEMS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7"]
WFG_PROBLEMS = ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

ZDT_POINTS = int(os.environ.get("VAMOS_REF_ZDT_POINTS", "200000"))
WFG_POINTS = int(os.environ.get("VAMOS_REF_WFG_POINTS", "200000"))
# WFG2 is particularly sensitive for HV normalization; allow a higher density.
WFG2_POINTS = int(os.environ.get("VAMOS_REF_WFG2_POINTS", str(WFG_POINTS)))
# WFG8 has a custom Pareto-set mapping; 200k points is already dense for stable HV normalization.
WFG8_POINTS = int(os.environ.get("VAMOS_REF_WFG8_POINTS", "200000"))
DTLZ_PARTITIONS = int(os.environ.get("VAMOS_REF_DTLZ_PARTITIONS", "700"))
DTLZ7_GRID = int(os.environ.get("VAMOS_REF_DTLZ7_GRID", "500"))

DTLZ_N_OBJ = 3
DTLZ7_G = 1.0
WFG_N_VAR = 24
WFG_N_OBJ = 2
WFG_SEED = int(os.environ.get("VAMOS_REF_WFG_SEED", "123"))


def _ensure_dirs() -> None:
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_front(problem: str, front: np.ndarray) -> None:
    name = problem.upper() if problem.lower().startswith("zdt") else problem.lower()
    out_dirs = [DATA_OUT_DIR]
    if problem.lower().startswith("zdt"):
        out_dirs.append(PACKAGE_OUT_DIR)
    for out_dir in out_dirs:
        path = out_dir / f"{name}.csv"
        np.savetxt(path, front, delimiter=",")


def _pareto_filter_2d(front: np.ndarray) -> np.ndarray:
    order = np.argsort(front[:, 0], kind="mergesort")
    sorted_front = front[order]
    keep = np.empty(sorted_front.shape[0], dtype=bool)
    best_f2 = np.inf
    for idx, f2 in enumerate(sorted_front[:, 1]):
        if f2 < best_f2:
            keep[idx] = True
            best_f2 = f2
        else:
            keep[idx] = False
    return sorted_front[keep]


def _zdt_front(problem: str, n_points: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=n_points)
    if problem == "zdt1":
        f1 = t
        f2 = 1.0 - np.sqrt(t)
    elif problem == "zdt2":
        f1 = t
        f2 = 1.0 - np.square(t)
    elif problem == "zdt3":
        f1 = t
        f2 = 1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * t)
    elif problem == "zdt4":
        f1 = t
        f2 = 1.0 - np.sqrt(t)
    elif problem == "zdt6":
        f1 = 1.0 - np.exp(-4.0 * t) * np.power(np.sin(6.0 * np.pi * t), 6)
        f2 = 1.0 - np.square(f1)
    else:
        raise ValueError(f"Unsupported ZDT problem '{problem}'.")
    front = np.column_stack([f1, f2])
    return _pareto_filter_2d(front)


def _dtlz_ref_dirs(n_partitions: int) -> np.ndarray:
    from pymoo.util.ref_dirs import get_reference_directions

    return get_reference_directions("das-dennis", DTLZ_N_OBJ, n_partitions=n_partitions)


def _dtlz_front(problem: str, n_partitions: int, dtlz7_grid: int) -> np.ndarray:
    if problem == "dtlz7":
        grid = np.linspace(0.0, 1.0, dtlz7_grid + 1)
        f1, f2 = np.meshgrid(grid, grid, indexing="xy")
        f1 = f1.ravel()
        f2 = f2.ravel()
        pair = np.column_stack([f1, f2])
        h = DTLZ_N_OBJ - np.sum(
            (pair / (1.0 + DTLZ7_G)) * (1.0 + np.sin(3.0 * np.pi * pair)),
            axis=1,
        )
        f3 = (1.0 + DTLZ7_G) * h
        return np.column_stack([f1, f2, f3])

    ref_dirs = _dtlz_ref_dirs(n_partitions)
    if problem == "dtlz1":
        return 0.5 * ref_dirs
    if problem in {"dtlz2", "dtlz3", "dtlz4"}:
        norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return ref_dirs / norms
    raise ValueError(f"Unsupported DTLZ problem '{problem}'.")


def _wfg_extremes(k: int) -> np.ndarray:
    count = 1 << k
    extremes = np.ones((count, k), dtype=float)
    for idx in range(count):
        for bit in range(k):
            if idx & (1 << bit):
                extremes[idx, bit] = 0.0
    return extremes


def _wfg_front(problem: str, n_points: int) -> np.ndarray:
    prob = get_problem(problem, n_var=WFG_N_VAR, n_obj=WFG_N_OBJ)
    k = int(prob.k)

    extremes = _wfg_extremes(k)
    if n_points <= extremes.shape[0]:
        K = extremes[:n_points]
    else:
        rng = np.random.default_rng(WFG_SEED)
        interior = rng.random((n_points - extremes.shape[0], k))
        K = np.vstack([extremes, interior])

    # Use the problem's own Pareto-set mapping to respect overrides (notably WFG8/WFG9).
    X = prob._positional_to_optimal(K)
    F = np.asarray(prob.evaluate(X, return_values_of=["F"]), dtype=float)
    return _pareto_filter_2d(F)


def main() -> None:
    _ensure_dirs()

    for name in ZDT_PROBLEMS:
        front = _zdt_front(name, ZDT_POINTS)
        _save_front(name, front)

    for name in DTLZ_PROBLEMS:
        front = _dtlz_front(name, DTLZ_PARTITIONS, DTLZ7_GRID)
        _save_front(name, front)

    for name in WFG_PROBLEMS:
        if name == "wfg2":
            points = WFG2_POINTS
        elif name == "wfg8":
            points = WFG8_POINTS
        else:
            points = WFG_POINTS
        front = _wfg_front(name, points)
        _save_front(name, front)


if __name__ == "__main__":
    main()

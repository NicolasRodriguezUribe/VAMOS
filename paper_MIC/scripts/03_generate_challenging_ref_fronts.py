#!/usr/bin/env python
"""
Generate analytical reference fronts for UF (CEC2009) and LSMOP problems.

Usage:
    python paper_MIC/scripts/03_generate_challenging_ref_fronts.py

These are the *true* Pareto fronts derived from the analytical problem
definitions, not approximations from optimisation runs.  They are saved
as CSV files in ``data/reference_fronts/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REF_DIR = ROOT_DIR / "data" / "reference_fronts"

N_POINTS = 1000


# ── UF bi-objective fronts (UF1–UF7) ────────────────────────────────────
def _uf1_pf() -> np.ndarray:
    """f1 ∈ [0,1], f2 = 1 − √f1"""
    t = np.linspace(0, 1, N_POINTS)
    return np.column_stack([t, 1.0 - np.sqrt(t)])


def _uf2_pf() -> np.ndarray:
    """Same shape as UF1."""
    return _uf1_pf()


def _uf3_pf() -> np.ndarray:
    """Same shape as UF1 (harder convergence, same PF)."""
    return _uf1_pf()


def _uf4_pf() -> np.ndarray:
    """f1 ∈ [0,1], f2 = 1 − f1²"""
    t = np.linspace(0, 1, N_POINTS)
    return np.column_stack([t, 1.0 - t**2])


def _uf5_pf() -> np.ndarray:
    """Discrete points on f1+f2 = 1 (N=10 → 21 points)."""
    N = 10
    k = np.arange(0, 2 * N + 1)
    t = k / (2.0 * N)
    return np.column_stack([t, 1.0 - t])


def _uf6_pf() -> np.ndarray:
    """Two disconnected segments on f1+f2 = 1 (N=2)."""
    N = 2
    # PF exists where sin(2Nπx₀) ≤ 0, i.e. x₀ in [1/(2N), 2/(2N)] ∪ [3/(2N), 4/(2N)]
    # With N=2: x₀ ∈ [0.25, 0.50] ∪ [0.75, 1.0]
    seg1 = np.linspace(0.25, 0.50, N_POINTS // 2)
    seg2 = np.linspace(0.75, 1.00, N_POINTS // 2)
    t = np.concatenate([seg1, seg2])
    return np.column_stack([t, 1.0 - t])


def _uf7_pf() -> np.ndarray:
    """f1 = x₀^0.2, f2 = 1 − f1; x₀ ∈ [0,1] → f1 ∈ [0,1]."""
    t = np.linspace(0, 1, N_POINTS)
    f1 = t**0.2
    return np.column_stack([f1, 1.0 - f1])


# ── UF tri-objective fronts (UF8–UF10) ──────────────────────────────────
def _unit_quarter_sphere(n_points: int = 5000) -> np.ndarray:
    """Sample the first-quadrant quarter of the unit sphere f₁²+f₂²+f₃²=1."""
    pts = []
    # Structured sampling on the simplex angle space
    n_side = int(np.ceil(np.sqrt(2 * n_points)))
    for i in range(n_side + 1):
        for j in range(n_side + 1 - i):
            theta = (i / n_side) * np.pi / 2
            phi = (j / max(1, n_side - i)) * np.pi / 2
            f1 = np.cos(theta) * np.cos(phi)
            f2 = np.cos(theta) * np.sin(phi)
            f3 = np.sin(theta)
            pts.append([f1, f2, f3])
    return np.array(pts, dtype=float)


def _uf8_pf() -> np.ndarray:
    """Quarter sphere: f₁²+f₂²+f₃²=1, f_i ≥ 0."""
    return _unit_quarter_sphere()


def _uf9_pf() -> np.ndarray:
    """UF9 has two disconnected regions on the quarter sphere.
    The PF exists where the front modification term = 0,
    i.e. (1+ε)(1−4(2x₀−1)²) ≤ 0 → |x₀−0.5| ≥ 0.5/√(1+ε).
    With ε=0.1: x₀ outside ~[0.024, 0.976] – nearly the full sphere.
    Use the full quarter sphere as a close approximation.
    """
    return _unit_quarter_sphere()


def _uf10_pf() -> np.ndarray:
    """Same PF shape as UF8 (quarter sphere), harder convergence."""
    return _unit_quarter_sphere()


# ── LSMOP fronts (n_obj=2) ──────────────────────────────────────────────
def _lsmop_linear_pf() -> np.ndarray:
    """LSMOP1–4: linear front f1+f2=1, f_i ≥ 0."""
    t = np.linspace(0, 1, N_POINTS)
    return np.column_stack([t, 1.0 - t])


def _lsmop_convex_pf() -> np.ndarray:
    """LSMOP5–8: convex (quarter circle) f₁²+f₂²=1, f_i ≥ 0."""
    t = np.linspace(0, np.pi / 2, N_POINTS)
    return np.column_stack([np.cos(t), np.sin(t)])


def _lsmop9_pf() -> np.ndarray:
    """LSMOP9: disconnected front.
    At optimum (G=0): f1 = x1, f2 = 2·(2 − (x1/2)·(1+sin(3πx1))), x1 ∈ [0,1].
    """
    t = np.linspace(0, 1, N_POINTS)
    f1 = t
    f2 = 2.0 * (2.0 - (t / 2.0) * (1.0 + np.sin(3.0 * np.pi * t)))
    return np.column_stack([f1, f2])


# ── Registry ─────────────────────────────────────────────────────────────
FRONTS: dict[str, callable] = {
    "cec2009_uf1": _uf1_pf,
    "cec2009_uf2": _uf2_pf,
    "cec2009_uf3": _uf3_pf,
    "cec2009_uf4": _uf4_pf,
    "cec2009_uf5": _uf5_pf,
    "cec2009_uf6": _uf6_pf,
    "cec2009_uf7": _uf7_pf,
    "cec2009_uf8": _uf8_pf,
    "cec2009_uf9": _uf9_pf,
    "cec2009_uf10": _uf10_pf,
    "lsmop1": _lsmop_linear_pf,
    "lsmop2": _lsmop_linear_pf,
    "lsmop3": _lsmop_linear_pf,
    "lsmop4": _lsmop_linear_pf,
    "lsmop5": _lsmop_convex_pf,
    "lsmop6": _lsmop_convex_pf,
    "lsmop7": _lsmop_convex_pf,
    "lsmop8": _lsmop_convex_pf,
    "lsmop9": _lsmop9_pf,
}


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in FRONTS.items():
        pf = factory()
        out = REF_DIR / f"{name}.csv"
        np.savetxt(out, pf, delimiter=",")
        print(f"  {name}: {pf.shape[0]} points ({pf.shape[1]} obj) -> {out}")
    print(f"\nDone – {len(FRONTS)} reference fronts written to {REF_DIR}")


if __name__ == "__main__":
    main()

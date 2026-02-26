from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_native_core():
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        candidates = [
            os.environ.get("VAMOSPP_DLL_DIR"),
            str(Path(sys.executable).resolve().parent),
        ]
        for tool in ("g++", "gcc", "clang++"):
            tool_path = shutil.which(tool)
            if tool_path:
                candidates.append(str(Path(tool_path).resolve().parent))

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                os.add_dll_directory(str(path))
            except (FileNotFoundError, OSError):
                pass
    return pytest.importorskip("vamospp._core", exc_type=ImportError)


core = _load_native_core()
from vamospp import _fallback


def _setup_small_case(pop_size: int = 8, n_var: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.full((pop_size, n_var), 0.5, dtype=np.float64)
    F = np.ascontiguousarray(
        np.column_stack(
            (
                np.sum(X, axis=1),
                np.sum((X - 0.5) ** 2, axis=1),
            )
        ),
        dtype=np.float64,
    )
    xl = np.zeros(n_var, dtype=np.float64)
    xu = np.ones(n_var, dtype=np.float64)
    return X, F, xl, xu


def _config() -> dict[str, float | int]:
    return {
        "sbx_prob": 0.0,
        "sbx_eta": 20.0,
        "pm_prob": 0.0,
        "pm_eta": 20.0,
        "tournament_pressure": 2,
    }


def _eval_fn(X_off: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(X_off, dtype=np.float64)
    F = np.ascontiguousarray(
        np.column_stack(
            (
                np.sum(arr, axis=1),
                np.sum((arr - 0.5) ** 2, axis=1),
            )
        ),
        dtype=np.float64,
    )
    return {"F": F}


def test_native_and_fallback_nsga2_evolve_smallcase_parity() -> None:
    X, F, xl, xu = _setup_small_case()
    cfg = _config()

    x_native, f_native = core.nsga2_evolve(X, F, xl, xu, cfg, 3, 77, _eval_fn)
    x_fallback, f_fallback = _fallback.nsga2_evolve(X, F, xl, xu, cfg, 3, 77, _eval_fn)

    assert x_native.shape == x_fallback.shape
    assert f_native.shape == f_fallback.shape
    np.testing.assert_allclose(x_native, x_fallback, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(f_native, f_fallback, rtol=0.0, atol=1e-12)

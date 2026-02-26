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


def _small_case(pop_size: int = 10, n_var: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _config() -> dict[str, float]:
    return {
        "sbx_prob": 0.0,
        "sbx_eta": 20.0,
        "pm_prob": 0.0,
        "pm_eta": 20.0,
    }


def test_native_smsemoa_generate_offspring_matches_fallback_smallcase() -> None:
    X, F, xl, xu = _small_case()
    cfg = _config()

    native = core.smsemoa_generate_offspring(X, F, "tournament", 2, xl, xu, cfg, 999, None)
    fallback = _fallback.smsemoa_generate_offspring(X, F, "tournament", 2, xl, xu, cfg, 999, None)

    assert native.shape == fallback.shape
    np.testing.assert_allclose(native, fallback, rtol=0.0, atol=1e-12)


def test_native_spea2_generate_offspring_matches_fallback_smallcase() -> None:
    X, F, xl, xu = _small_case()
    cfg = _config()

    native = core.spea2_generate_offspring(X, F, 5, 2, xl, xu, cfg, 2026, None)
    fallback = _fallback.spea2_generate_offspring(X, F, 5, 2, xl, xu, cfg, 2026, None)

    assert native.shape == fallback.shape
    np.testing.assert_allclose(native, fallback, rtol=0.0, atol=1e-12)

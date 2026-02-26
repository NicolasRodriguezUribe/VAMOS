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


def _sample(pop_size: int = 10, n_var: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2024)
    X = rng.random((pop_size, n_var), dtype=np.float64)
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


def _config(n_var: int) -> dict[str, float | int]:
    return {
        "sbx_prob": 0.9,
        "sbx_eta": 20.0,
        "pm_prob": 1.0 / max(1, n_var),
        "pm_eta": 20.0,
        "tournament_pressure": 2,
    }


def test_generate_offspring_honors_out_buffer() -> None:
    X, F, xl, xu = _sample()
    n_offspring = X.shape[0]
    out = np.full((n_offspring, X.shape[1]), np.nan, dtype=np.float64)

    returned = core.generate_offspring(X, F, n_offspring, xl, xu, _config(X.shape[1]), 111, out)

    assert returned is out
    assert out.flags.c_contiguous
    assert np.isfinite(out).all()


def test_generate_offspring_out_shape_mismatch_raises() -> None:
    X, F, xl, xu = _sample()
    n_offspring = X.shape[0]
    out = np.empty((n_offspring + 1, X.shape[1]), dtype=np.float64)

    with pytest.raises(RuntimeError, match="out has wrong shape"):
        core.generate_offspring(X, F, n_offspring, xl, xu, _config(X.shape[1]), 222, out)

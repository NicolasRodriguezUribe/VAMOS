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


def _sample_population(pop_size: int = 14, n_var: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9102)
    X = rng.random((pop_size, n_var), dtype=np.float64)
    F = np.ascontiguousarray(
        np.column_stack(
            (
                np.sum(X, axis=1),
                np.sum((X - 0.25) ** 2, axis=1),
            )
        ),
        dtype=np.float64,
    )
    xl = np.zeros(n_var, dtype=np.float64)
    xu = np.ones(n_var, dtype=np.float64)
    return X, F, xl, xu


def _config(n_var: int) -> dict[str, float]:
    return {
        "sbx_prob": 0.9,
        "sbx_eta": 20.0,
        "pm_prob": 1.0 / max(1, n_var),
        "pm_eta": 20.0,
    }


def _assert_float64_c_2d(arr: np.ndarray, shape: tuple[int, int]) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    assert arr.shape == shape


def test_native_smsemoa_generate_offspring_ndarray_and_out_contract() -> None:
    X, F, xl, xu = _sample_population()
    cfg = _config(X.shape[1])

    child = core.smsemoa_generate_offspring(X, F, "tournament", 2, xl, xu, cfg, 123, None)
    _assert_float64_c_2d(child, (1, X.shape[1]))

    out = np.empty((1, X.shape[1]), dtype=np.float64)
    returned = core.smsemoa_generate_offspring(X, F, "tournament", 2, xl, xu, cfg, 123, out)
    assert returned is out
    _assert_float64_c_2d(out, (1, X.shape[1]))
    np.testing.assert_allclose(out, child, rtol=0.0, atol=0.0)

    bad_out = np.empty((2, X.shape[1]), dtype=np.float64)
    with pytest.raises(RuntimeError, match="out has wrong shape"):
        core.smsemoa_generate_offspring(X, F, "tournament", 2, xl, xu, cfg, 123, bad_out)


def test_native_spea2_generate_offspring_ndarray_out_and_empty_contract() -> None:
    X, F, xl, xu = _sample_population()
    cfg = _config(X.shape[1])
    n_offspring = 6

    offspring = core.spea2_generate_offspring(X, F, n_offspring, 2, xl, xu, cfg, 321, None)
    _assert_float64_c_2d(offspring, (n_offspring, X.shape[1]))

    out = np.empty((n_offspring, X.shape[1]), dtype=np.float64)
    returned = core.spea2_generate_offspring(X, F, n_offspring, 2, xl, xu, cfg, 321, out)
    assert returned is out
    _assert_float64_c_2d(out, (n_offspring, X.shape[1]))
    np.testing.assert_allclose(out, offspring, rtol=0.0, atol=0.0)

    empty = core.spea2_generate_offspring(X, F, 0, 2, xl, xu, cfg, 444, None)
    _assert_float64_c_2d(empty, (0, X.shape[1]))

    bad_out = np.empty((n_offspring + 1, X.shape[1]), dtype=np.float64)
    with pytest.raises(RuntimeError, match="out has wrong shape"):
        core.spea2_generate_offspring(X, F, n_offspring, 2, xl, xu, cfg, 321, bad_out)

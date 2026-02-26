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


def _sample_case(pop_size: int = 16, n_var: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260226)
    X = rng.random((pop_size, n_var), dtype=np.float64)
    F = np.ascontiguousarray(
        np.column_stack(
            (
                np.sum(X, axis=1),
                np.sum((X - 0.3) ** 2, axis=1),
            )
        ),
        dtype=np.float64,
    )
    xl = np.zeros(n_var, dtype=np.float64)
    xu = np.ones(n_var, dtype=np.float64)
    return X, F, xl, xu


def _assert_float64_c_2d(arr: np.ndarray, shape: tuple[int, int]) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous
    assert arr.shape == shape


def test_native_operator_matrix_kernels_return_ndarray_contracts() -> None:
    X, F, xl, xu = _sample_case()

    crossed = core.sbx_crossover(X[:7], 0.9, 20.0, xl, xu, 123, 0.5)
    _assert_float64_c_2d(crossed, (8, X.shape[1]))

    mutated = core.polynomial_mutation(X[:9], 0.2, 20.0, xl, xu, 987, False)
    _assert_float64_c_2d(mutated, (9, X.shape[1]))

    dom = core.dominance_matrix(F)
    assert isinstance(dom, np.ndarray)
    assert dom.dtype == np.bool_
    assert dom.flags.c_contiguous
    assert dom.shape == (F.shape[0], F.shape[0])

    fitness, dist = core.spea2_fitness(F, None, 2)
    assert isinstance(fitness, np.ndarray)
    assert fitness.dtype == np.float64
    assert fitness.flags.c_contiguous
    assert fitness.shape == (F.shape[0],)
    _assert_float64_c_2d(dist, (F.shape[0], F.shape[0]))

    ref = np.max(F, axis=0) + 1.0
    indicator = core.ibea_indicator_matrix(F, ref, "epsilon")
    _assert_float64_c_2d(indicator, (F.shape[0], F.shape[0]))


def test_native_spea2_fitness_and_ibea_indicator_match_fallback_smallcase() -> None:
    _, F, _, _ = _sample_case(pop_size=10, n_var=4)

    fit_native, dist_native = core.spea2_fitness(F, None, 2)
    fit_fallback, dist_fallback = _fallback.spea2_fitness(F, None, 2)
    np.testing.assert_allclose(fit_native, fit_fallback, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(dist_native, dist_fallback, rtol=0.0, atol=1e-12)

    ref = np.max(F, axis=0) + 1.0
    ind_native = core.ibea_indicator_matrix(F, ref, "epsilon")
    ind_fallback = _fallback.ibea_indicator_matrix(F, ref, "epsilon")
    np.testing.assert_allclose(ind_native, ind_fallback, rtol=0.0, atol=1e-12)

from __future__ import annotations

import numpy as np
import pytest


def _cpp_backend():
    pytest.importorskip("vamospp")
    try:
        from vamos.foundation.kernel.cpp_backend import CppBackend

        return CppBackend()
    except ImportError:
        pytest.skip("cpp kernel is not available")


def test_cpp_rejects_non_float64_ranking_input() -> None:
    cpp = _cpp_backend()
    F = np.ones((8, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="dtype float64"):
        cpp.nsga2_ranking(F)


def test_cpp_rejects_non_contiguous_ranking_input() -> None:
    cpp = _cpp_backend()
    F = np.ones((8, 2), dtype=np.float64)[::2]
    assert not F.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        cpp.nsga2_ranking(F)


def test_cpp_rejects_non_int64_tournament_ranks() -> None:
    cpp = _cpp_backend()
    ranks = np.arange(8, dtype=np.int32)
    crowd = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="dtype int64"):
        cpp.tournament_selection(ranks, crowd, pressure=2, rng=rng, n_parents=4)


def test_cpp_rejects_non_float64_bounds_in_fused_paths() -> None:
    cpp = _cpp_backend()
    rng = np.random.default_rng(1)
    X = rng.random((10, 3), dtype=np.float64)
    F = rng.random((10, 2), dtype=np.float64)
    params = {"sbx_prob": 0.9, "sbx_eta": 20.0, "pm_prob": 0.2, "pm_eta": 20.0}
    xl = np.zeros(3, dtype=np.float32)
    xu = np.ones(3, dtype=np.float64)
    with pytest.raises(TypeError, match="dtype float64"):
        cpp.spea2_generate_offspring(X, F, 5, 1, params, rng, xl, xu)

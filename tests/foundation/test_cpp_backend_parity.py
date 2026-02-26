from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.ibea.helpers import environmental_selection as ibea_environmental_selection
from vamos.engine.algorithm.spea2.helpers import environmental_selection as spea2_environmental_selection
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions


def _cpp_backend():
    pytest.importorskip("vamospp")
    try:
        from vamos.foundation.kernel.cpp_backend import CppBackend

        return CppBackend()
    except ImportError:
        pytest.skip("cpp kernel is not available")


def test_smsemoa_remove_index_matches_python_reference() -> None:
    cpp = _cpp_backend()
    np_kernel = NumPyKernel()
    rng = np.random.default_rng(0)
    F = rng.random((20, 2), dtype=np.float64)
    ref = np.max(F, axis=0) + 1.0

    ranks, _ = np_kernel.nsga2_ranking(F)
    worst_rank = int(ranks.max(initial=0))
    worst_idx = np.flatnonzero(ranks == worst_rank)
    if worst_idx.size == 1:
        expected = int(worst_idx[0])
    else:
        contribs = hypervolume_contributions(F[worst_idx], ref)
        expected = int(worst_idx[int(np.argmin(contribs))])

    actual = cpp.smsemoa_remove_index(F, ref)
    assert actual == expected


def test_spea2_environmental_selection_indices_match_python_reference() -> None:
    cpp = _cpp_backend()
    rng = np.random.default_rng(1)
    F = rng.random((24, 3), dtype=np.float64)
    X = np.arange(F.shape[0], dtype=np.float64).reshape(-1, 1)
    keep = 12

    sel_X, _, _ = spea2_environmental_selection(
        X,
        F,
        None,
        archive_size=keep,
        k_neighbors=2,
        constraint_mode="none",
        kernel=None,
    )
    expected_idx = sel_X[:, 0].astype(np.int64)
    actual_idx = cpp.spea2_environmental_selection_indices(F, keep, 2)
    np.testing.assert_array_equal(actual_idx, expected_idx)


def test_ibea_environmental_selection_indices_match_python_reference() -> None:
    cpp = _cpp_backend()
    rng = np.random.default_rng(2)
    F = rng.random((18, 2), dtype=np.float64)
    X = np.arange(F.shape[0], dtype=np.float64).reshape(-1, 1)
    keep = 10
    kappa = 0.05

    exp_X, _, _, exp_fit = ibea_environmental_selection(
        X,
        F,
        None,
        pop_size=keep,
        indicator="epsilon",
        kappa=kappa,
        kernel=None,
    )
    expected_idx = exp_X[:, 0].astype(np.int64)

    actual_idx, actual_fit = cpp.ibea_environmental_selection_indices(F, keep, None, "epsilon", kappa)
    np.testing.assert_array_equal(actual_idx, expected_idx)
    np.testing.assert_allclose(actual_fit, exp_fit, rtol=0.0, atol=1e-12)

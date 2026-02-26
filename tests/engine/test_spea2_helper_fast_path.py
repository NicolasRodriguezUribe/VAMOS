from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.spea2.helpers import environmental_selection


class _KernelStub:
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.calls: int = 0

    def spea2_environmental_selection_indices(self, F: np.ndarray, archive_size: int, k: int) -> np.ndarray:
        self.calls += 1
        assert F.ndim == 2
        assert archive_size == 2
        assert k == 1
        return self.indices


def test_environmental_selection_uses_kernel_fast_indices() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    F = np.array([[0.2, 0.9], [0.5, 0.5], [0.9, 0.2]], dtype=np.float64)
    kernel = _KernelStub(indices=np.array([0, 2], dtype=np.int64))

    sel_X, sel_F, sel_G = environmental_selection(
        X,
        F,
        None,
        archive_size=2,
        k_neighbors=1,
        constraint_mode="none",
        kernel=kernel,
    )

    assert kernel.calls == 1
    np.testing.assert_allclose(sel_X, X[[0, 2]])
    np.testing.assert_allclose(sel_F, F[[0, 2]])
    assert sel_G is None

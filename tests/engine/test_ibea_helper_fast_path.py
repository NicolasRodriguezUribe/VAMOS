from __future__ import annotations

import numpy as np

from vamos.engine.algorithm.ibea.helpers import environmental_selection


class _KernelStub:
    def __init__(self, indices: np.ndarray, fitness: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.fitness = np.asarray(fitness, dtype=np.float64)
        self.calls: int = 0

    def ibea_environmental_selection_indices(
        self,
        F: np.ndarray,
        pop_size: int,
        reference_point: np.ndarray | None,
        kind: str,
        kappa: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        assert F.ndim == 2
        assert pop_size == 2
        assert reference_point is None
        assert kind == "epsilon"
        assert float(kappa) == 1.0
        return self.indices, self.fitness


def test_environmental_selection_uses_kernel_fast_indices() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    F = np.array([[0.2, 0.9], [0.5, 0.5], [0.9, 0.2]], dtype=np.float64)
    kernel = _KernelStub(indices=np.array([0, 2], dtype=np.int64), fitness=np.array([-0.8, -0.3], dtype=np.float64))

    sel_X, sel_F, sel_G, fitness = environmental_selection(
        X,
        F,
        None,
        pop_size=2,
        indicator="epsilon",
        kappa=1.0,
        kernel=kernel,
    )

    assert kernel.calls == 1
    np.testing.assert_allclose(sel_X, X[[0, 2]])
    np.testing.assert_allclose(sel_F, F[[0, 2]])
    np.testing.assert_allclose(fitness, np.array([-0.8, -0.3], dtype=np.float64))
    assert sel_G is None

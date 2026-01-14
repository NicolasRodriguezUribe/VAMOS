from __future__ import annotations

import time

import numpy as np
import pytest

from vamos.foundation.kernel.numpy_backend import NumPyKernel


@pytest.mark.slow
def test_numpy_nsga2_ranking_perf_smoke() -> None:
    rng = np.random.default_rng(0)
    F = rng.random((400, 3))
    kernel = NumPyKernel()

    start = time.perf_counter()
    ranks, crowding = kernel.nsga2_ranking(F)
    elapsed = time.perf_counter() - start

    assert ranks.shape == (F.shape[0],)
    assert crowding.shape == (F.shape[0],)
    assert elapsed < 5.0

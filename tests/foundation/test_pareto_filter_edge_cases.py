from __future__ import annotations

import numpy as np

from vamos.foundation.metrics.pareto import pareto_filter


def test_pareto_filter_keeps_duplicates() -> None:
    F = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    front, idx = pareto_filter(F, return_indices=True)

    assert front.shape[0] == 2
    assert set(idx.tolist()) == {0, 1}


def test_pareto_filter_handles_none() -> None:
    front, idx = pareto_filter(None, return_indices=True)
    assert front.shape == (0, 0)
    assert idx.size == 0

from __future__ import annotations

import numpy as np

import vamos.foundation.kernel.numpy_backend as numpy_backend


class _ChoiceForbiddenRNG:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def random(self, size: tuple[int, int]) -> np.ndarray:
        return self._rng.random(size)

    def choice(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("choice() should not be used in the chunked tournament path.")


def test_chunked_tournament_sampling_avoids_rng_choice(monkeypatch) -> None:
    monkeypatch.setattr(numpy_backend, "_TOURNAMENT_KEY_BUDGET", 8)
    rng = _ChoiceForbiddenRNG(seed=4)

    candidates = numpy_backend._sample_tournament_candidates(
        rng,
        n_candidates=9,
        n_parents=6,
        pressure=4,
    )

    assert candidates.shape == (6, 4)
    for row in candidates:
        assert len(set(row.tolist())) == 4


def test_blocked_non_dominated_sort_matches_dense_reference(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    F = rng.integers(0, 9, size=(18, 3)).astype(float)

    dense_fronts, dense_ranks = numpy_backend._fast_non_dominated_sort_dense(F)
    monkeypatch.setattr(numpy_backend, "_DOMINANCE_TENSOR_BUDGET", 10)
    blocked_fronts, blocked_ranks = numpy_backend._fast_non_dominated_sort(F)

    assert blocked_fronts == dense_fronts
    np.testing.assert_array_equal(blocked_ranks, dense_ranks)

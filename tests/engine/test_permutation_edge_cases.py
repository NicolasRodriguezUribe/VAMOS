from __future__ import annotations

import numpy as np

from vamos.engine.operators.impl._permutation_common import ensure_distinct_indices
from vamos.engine.operators.impl._permutation_crossovers import order_crossover
from vamos.engine.operators.impl._permutation_mutations import inversion_mutation, scramble_mutation, swap_mutation


class _FixedRng:
    def __init__(self, integers: list[int]) -> None:
        self._integers = list(integers)

    def random(self, size=None):  # noqa: ANN001
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)

    def integers(self, low, high=None, size=None):  # noqa: ANN001
        if size is not None:
            return np.zeros(size, dtype=int)
        return self._integers.pop(0)

    def shuffle(self, values: np.ndarray) -> None:
        values[:] = values[::-1]


def test_ensure_distinct_indices_noops_when_only_one_index_exists() -> None:
    idx = np.array([[0, 0]], dtype=int)

    ensure_distinct_indices(idx, upper=1, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(idx, np.array([[0, 0]], dtype=int))


def test_order_crossover_noops_for_single_gene_permutation() -> None:
    parents = np.array([[0], [0]], dtype=int)

    offspring = order_crossover(parents, prob=1.0, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(offspring, parents)
    assert offspring is not parents


def test_swap_mutation_noops_for_single_gene_permutation() -> None:
    X = np.array([[0], [0]], dtype=int)

    swap_mutation(X, prob=1.0, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(X, np.array([[0], [0]], dtype=int))


def test_scramble_mutation_zero_segment_limit_means_unbounded() -> None:
    X = np.arange(60, dtype=int)[None, :]

    scramble_mutation(X, prob=1.0, rng=_FixedRng([0, 59]), max_segment_length=0)

    np.testing.assert_array_equal(X[0], np.arange(59, -1, -1, dtype=int))


def test_inversion_mutation_can_include_last_gene() -> None:
    X = np.arange(5, dtype=int)[None, :]

    inversion_mutation(X, prob=1.0, rng=_FixedRng([0, 3]))

    np.testing.assert_array_equal(X[0], np.array([4, 3, 2, 1, 0], dtype=int))

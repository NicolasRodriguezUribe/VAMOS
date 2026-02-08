import numpy as np

from vamos.operators.impl.binary import (
    random_binary_population,
    one_point_crossover,
    bit_flip_mutation,
)
from vamos.operators.impl.integer import (
    random_integer_population,
    uniform_integer_crossover,
    random_reset_mutation,
)
from vamos.operators.impl.mixed import mixed_initialize, mixed_crossover, mixed_mutation


def test_binary_operators_shape_and_values():
    rng = np.random.default_rng(0)
    X = random_binary_population(5, 8, rng)
    assert X.shape == (5, 8)
    assert set(np.unique(X)).issubset({0, 1})

    parents = np.vstack([X, X]).reshape(-1, 2, 8)
    children = one_point_crossover(parents.reshape(-1, 8), prob=1.0, rng=rng)
    assert children.shape == parents.reshape(-1, 8).shape

    pre = children.copy()
    bit_flip_mutation(children, prob=1.0, rng=rng)
    assert not np.array_equal(children, pre)
    assert set(np.unique(children)).issubset({0, 1})


def test_integer_operators_respect_bounds():
    rng = np.random.default_rng(1)
    lower = np.array([0, 5, 10])
    upper = np.array([3, 7, 12])
    X = random_integer_population(4, 3, lower, upper, rng)
    assert X.shape == (4, 3)
    assert np.all(X >= lower) and np.all(X <= upper)

    parents = np.vstack([X, X]).reshape(-1, 2, 3)
    children = uniform_integer_crossover(parents.reshape(-1, 3), prob=1.0, rng=rng)
    assert children.shape == parents.reshape(-1, 3).shape

    random_reset_mutation(children, prob=1.0, lower=lower, upper=upper, rng=rng)
    assert np.all(children >= lower) and np.all(children <= upper)


def test_mixed_initialize_and_variation():
    rng = np.random.default_rng(2)
    spec = {
        "real_idx": np.array([0, 1]),
        "int_idx": np.array([2]),
        "cat_idx": np.array([3]),
        "real_lower": np.array([0.0, -1.0]),
        "real_upper": np.array([1.0, 1.0]),
        "int_lower": np.array([0]),
        "int_upper": np.array([3]),
        "cat_cardinality": np.array([4]),
    }
    X = mixed_initialize(6, 4, spec, rng)
    assert X.shape == (6, 4)
    # Real within bounds
    assert np.all(X[:, spec["real_idx"]] >= spec["real_lower"])
    assert np.all(X[:, spec["real_idx"]] <= spec["real_upper"])
    # Int within bounds
    assert np.all(X[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(X[:, spec["int_idx"]] <= spec["int_upper"])
    # Categories within range
    assert np.all((X[:, spec["cat_idx"]] >= 0) & (X[:, spec["cat_idx"]] < spec["cat_cardinality"]))

    parents = np.vstack([X[:4], X[:4]]).reshape(-1, 2, 4)
    children = mixed_crossover(parents.reshape(-1, 4), prob=1.0, spec=spec, rng=rng)
    assert children.shape == parents.reshape(-1, 4).shape
    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    assert np.all(children[:, spec["real_idx"]] >= spec["real_lower"])
    assert np.all(children[:, spec["real_idx"]] <= spec["real_upper"])
    assert np.all(children[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(children[:, spec["int_idx"]] <= spec["int_upper"])
    assert np.all((children[:, spec["cat_idx"]] >= 0) & (children[:, spec["cat_idx"]] < spec["cat_cardinality"]))


def test_mixed_with_permutation_segment():
    rng = np.random.default_rng(3)
    perm_idx = np.array([0, 1, 2, 3, 4])
    spec = {
        "perm_idx": perm_idx,
        "int_idx": np.array([5]),
        "real_idx": np.array([6, 7]),
        "real_lower": np.array([-1.0, 0.0]),
        "real_upper": np.array([1.0, 2.0]),
        "int_lower": np.array([1]),
        "int_upper": np.array([3]),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
    }
    n_var = 8
    X = mixed_initialize(6, n_var, spec, rng)
    assert X.shape == (6, n_var)

    expected = set(range(perm_idx.size))
    for row in X[:, perm_idx]:
        assert set(row.astype(int)) == expected

    parents = np.vstack([X[:4], X[:4]]).reshape(-1, 2, n_var)
    children = mixed_crossover(parents.reshape(-1, n_var), prob=1.0, spec=spec, rng=rng)
    assert children.shape == parents.reshape(-1, n_var).shape

    for row in children[:, perm_idx]:
        assert set(row.astype(int)) == expected

    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    for row in children[:, perm_idx]:
        assert set(row.astype(int)) == expected
    assert np.all(children[:, spec["int_idx"]] >= spec["int_lower"])
    assert np.all(children[:, spec["int_idx"]] <= spec["int_upper"])


def test_mixed_segment_specific_probabilities_override_global_probability():
    rng = np.random.default_rng(7)
    perm_idx = np.array([0, 1, 2, 3], dtype=int)
    int_idx = np.array([4, 5], dtype=int)
    spec = {
        "perm_idx": perm_idx,
        "int_idx": int_idx,
        "int_lower": np.array([0, 0], dtype=int),
        "int_upper": np.array([9, 9], dtype=int),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
        "perm_crossover_prob": 0.0,
        "int_crossover_prob": 1.0,
        "perm_mutation_prob": 0.0,
        "int_mutation_prob": 1.0,
        "int_mutation": "reset",
    }
    parents = np.array(
        [
            [0, 1, 2, 3, 1, 8],
            [3, 2, 1, 0, 7, 2],
            [1, 0, 3, 2, 2, 6],
            [2, 3, 0, 1, 8, 1],
        ],
        dtype=float,
    )
    children = mixed_crossover(parents, prob=0.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, perm_idx], parents[:, perm_idx])
    assert np.any(children[:, int_idx] != parents[:, int_idx])

    before_mutation = children.copy()
    mixed_mutation(children, prob=0.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, perm_idx], before_mutation[:, perm_idx])
    assert np.any(children[:, int_idx] != before_mutation[:, int_idx])
    assert np.all(children[:, int_idx] >= spec["int_lower"])
    assert np.all(children[:, int_idx] <= spec["int_upper"])


def test_mixed_integer_segment_supports_arithmetic_crossover_and_creep_mutation():
    rng = np.random.default_rng(11)
    spec = {
        "perm_idx": np.array([0, 1, 2], dtype=int),
        "int_idx": np.array([3, 4], dtype=int),
        "int_lower": np.array([0, 0], dtype=int),
        "int_upper": np.array([10, 10], dtype=int),
        "perm_crossover": "ox",
        "perm_mutation": "swap",
        "perm_crossover_prob": 0.0,
        "int_crossover_prob": 1.0,
        "int_crossover": "arithmetic",
        "perm_mutation_prob": 0.0,
        "int_mutation_prob": 1.0,
        "int_mutation": "creep",
        "int_mutation_step": 2,
    }
    parents = np.array(
        [
            [0, 1, 2, 2, 8],
            [2, 1, 0, 8, 2],
        ],
        dtype=float,
    )
    children = mixed_crossover(parents, prob=1.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, :3], parents[:, :3])
    assert np.array_equal(children[:, 3:], np.array([[5.0, 5.0], [5.0, 5.0]]))

    before_mutation = children.copy()
    mixed_mutation(children, prob=1.0, spec=spec, rng=rng)
    assert np.array_equal(children[:, :3], before_mutation[:, :3])
    deltas = np.abs(children[:, 3:] - before_mutation[:, 3:])
    assert np.array_equal(deltas, np.full_like(deltas, 2.0))

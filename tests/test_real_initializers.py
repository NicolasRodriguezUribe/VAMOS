import numpy as np

from vamos.operators.real.initialize import LatinHypercubeInitializer, ScatterSearchInitializer


def test_latin_hypercube_initializer_bounds_and_shape():
    rng = np.random.default_rng(1)
    lower = np.zeros(3)
    upper = np.ones(3)
    init = LatinHypercubeInitializer(10, lower, upper, rng=rng)
    X = init()
    assert X.shape == (10, 3)
    assert np.all(X >= lower) and np.all(X <= upper)


def test_scatter_search_initializer_uses_base_and_children():
    rng = np.random.default_rng(2)
    lower = np.zeros(2)
    upper = np.ones(2)
    init = ScatterSearchInitializer(15, lower, upper, base_size=5, rng=rng)
    X = init()
    assert X.shape == (15, 2)
    assert np.all(X >= lower) and np.all(X <= upper)

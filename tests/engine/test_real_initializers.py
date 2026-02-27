import numpy as np
import pytest

from vamos.engine.operators.impl.real.initialize import (
    HaltonInitializer,
    LatinHypercubeInitializer,
    OppositionBasedInitializer,
    ScatterSearchInitializer,
    SobolInitializer,
)


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


# --- Sobol ---


def test_sobol_initializer_bounds_and_shape():
    rng = np.random.default_rng(42)
    lower = np.zeros(4)
    upper = np.ones(4)
    init = SobolInitializer(20, lower, upper, rng=rng)
    X = init()
    assert X.shape == (20, 4)
    assert np.all(X >= lower) and np.all(X <= upper)


def test_sobol_initializer_non_power_of_two():
    rng = np.random.default_rng(7)
    lower = np.array([0.0, 0.0])
    upper = np.array([10.0, 10.0])
    init = SobolInitializer(13, lower, upper, rng=rng)
    X = init()
    assert X.shape == (13, 2)
    assert np.all(X >= lower) and np.all(X <= upper)


def test_sobol_initializer_no_scramble():
    rng = np.random.default_rng(1)
    lower = np.zeros(3)
    upper = np.ones(3)
    init = SobolInitializer(8, lower, upper, rng=rng, scramble=False)
    X = init()
    assert X.shape == (8, 3)
    assert np.all(X >= lower) and np.all(X <= upper)


# --- Halton ---


def test_halton_initializer_bounds_and_shape():
    rng = np.random.default_rng(42)
    lower = np.zeros(3)
    upper = np.ones(3)
    init = HaltonInitializer(10, lower, upper, rng=rng)
    X = init()
    assert X.shape == (10, 3)
    assert np.all(X >= lower) and np.all(X <= upper)


def test_halton_initializer_scaled_bounds():
    rng = np.random.default_rng(5)
    lower = np.array([-5.0, 0.0])
    upper = np.array([5.0, 100.0])
    init = HaltonInitializer(25, lower, upper, rng=rng)
    X = init()
    assert X.shape == (25, 2)
    assert np.all(X >= lower) and np.all(X <= upper)


# --- Opposition-Based Learning ---


class _MockSingleObjProblem:
    n_var = 2
    n_obj = 1

    def evaluate(self, X, out):
        out["F"][:, 0] = np.sum(X**2, axis=1)


class _MockMultiObjProblem:
    n_var = 2
    n_obj = 2

    def evaluate(self, X, out):
        out["F"][:, 0] = X[:, 0]
        out["F"][:, 1] = (1.0 + X[:, 1]) / (X[:, 0] + 1e-9)


def test_obl_initializer_single_objective():
    rng = np.random.default_rng(99)
    lower = np.array([-5.0, -5.0])
    upper = np.array([5.0, 5.0])
    problem = _MockSingleObjProblem()
    init = OppositionBasedInitializer(10, lower, upper, problem=problem, rng=rng)
    X = init()
    assert X.shape == (10, 2)
    assert np.all(X >= lower) and np.all(X <= upper)


def test_obl_initializer_multi_objective():
    rng = np.random.default_rng(123)
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])
    problem = _MockMultiObjProblem()
    init = OppositionBasedInitializer(8, lower, upper, problem=problem, rng=rng)
    X = init()
    assert X.shape == (8, 2)
    assert np.all(X >= lower) and np.all(X <= upper)


# --- Dispatcher integration ---

from vamos.engine.algorithm.components.population import initialize_population


def test_dispatcher_sobol():
    rng = np.random.default_rng(1)
    xl = np.zeros(3)
    xu = np.ones(3)
    X = initialize_population(10, 3, xl, xu, "real", rng, initializer={"type": "sobol"})
    assert X.shape == (10, 3)


def test_dispatcher_halton():
    rng = np.random.default_rng(1)
    xl = np.zeros(3)
    xu = np.ones(3)
    X = initialize_population(10, 3, xl, xu, "real", rng, initializer={"type": "halton"})
    assert X.shape == (10, 3)


def test_dispatcher_obl():
    rng = np.random.default_rng(1)
    xl = np.zeros(2)
    xu = np.ones(2)
    problem = _MockSingleObjProblem()
    X = initialize_population(10, 2, xl, xu, "real", rng, problem=problem, initializer={"type": "obl"})
    assert X.shape == (10, 2)


def test_dispatcher_obl_no_problem_raises():
    rng = np.random.default_rng(1)
    xl = np.zeros(2)
    xu = np.ones(2)
    with pytest.raises(ValueError, match="requires a problem"):
        initialize_population(10, 2, xl, xu, "real", rng, problem=None, initializer={"type": "obl"})

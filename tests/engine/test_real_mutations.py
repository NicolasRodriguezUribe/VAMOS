import numpy as np

from vamos.operators.impl.real.mutation import UniformMutation, LinkedPolynomialMutation
from vamos.operators.impl.real.repair import ClampRepair


def test_uniform_mutation_respects_bounds_with_repair():
    rng = np.random.default_rng(3)
    lower = np.zeros(4)
    upper = np.ones(4)
    mut = UniformMutation(prob=1.0, perturb=0.2, lower=lower, upper=upper, repair=ClampRepair(), rng=rng)
    x = np.full(4, 0.5)
    y = mut(x)
    assert np.all(y >= lower) and np.all(y <= upper)


def test_linked_polynomial_mutation_shared_delta():
    rng = np.random.default_rng(4)
    lower = np.zeros(3)
    upper = np.ones(3)
    mut = LinkedPolynomialMutation(prob=1.0, eta=20.0, lower=lower, upper=upper, repair=None, rng=rng)
    x = np.array([0.2, 0.4, 0.6])
    y = mut(x)
    deltas = (y - x) / (upper - lower)
    # All deltas (where mutation applied) should be equal
    assert np.allclose(deltas, deltas[0])


def test_uniform_mutation_handles_population_input():
    rng = np.random.default_rng(5)
    lower = np.zeros(3)
    upper = np.ones(3)
    mut = UniformMutation(prob=1.0, perturb=0.3, lower=lower, upper=upper, repair=ClampRepair(), rng=rng)
    x = np.full((2, 3), 0.5)
    y = mut(x)
    assert y.shape == x.shape
    assert np.all(y >= lower) and np.all(y <= upper)


def test_linked_polynomial_mutation_handles_population_input():
    rng = np.random.default_rng(6)
    lower = np.zeros(2)
    upper = np.ones(2)
    mut = LinkedPolynomialMutation(prob=1.0, eta=15.0, lower=lower, upper=upper, repair=None, rng=rng)
    x = np.full((3, 2), 0.4)
    y = mut(x)
    assert y.shape == x.shape
    assert np.all(y >= lower) and np.all(y <= upper)

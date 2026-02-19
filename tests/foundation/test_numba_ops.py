import numpy as np
import pytest

from vamos.foundation.kernel.numba_ops import polynomial_mutation_numba, sbx_crossover_numba


@pytest.mark.numba
def test_polynomial_mutation_numba_bounds():
    """Verify mutation respects bounds."""
    N, D = 100, 5
    X = np.random.rand(N, D)
    lower = np.zeros(D)
    upper = np.ones(D)

    # Mutate with high probability
    polynomial_mutation_numba(X, 1.0, 20.0, lower, upper)

    assert np.all(X >= lower)
    assert np.all(X <= upper)


@pytest.mark.numba
def test_polynomial_mutation_numba_no_change_zero_prob():
    """Verify prob=0.0 causes no changes."""
    N, D = 100, 5
    X_orig = np.random.rand(N, D)
    X = X_orig.copy()
    lower = np.zeros(D)
    upper = np.ones(D)

    polynomial_mutation_numba(X, 0.0, 20.0, lower, upper)

    np.testing.assert_array_equal(X, X_orig)


@pytest.mark.numba
def test_sbx_crossover_numba_bounds():
    """Verify crossover respects bounds."""
    N, D = 100, 5
    parents = np.random.rand(N, D)
    lower = np.zeros(D)
    upper = np.ones(D)

    # We pass parents directly; Numba backend handles reshaping but test passes flat
    # Actually our implementation expects flatten-ish usage but handles pairs internally
    offspring = sbx_crossover_numba(parents, 1.0, 20.0, lower, upper)

    assert offspring.shape == parents.shape
    assert np.all(offspring >= lower)
    assert np.all(offspring <= upper)


@pytest.mark.numba
def test_sbx_crossover_numba_change():
    """Verify prob=1.0 causes changes."""
    N, D = 100, 5
    parents = np.random.rand(N, D)
    lower = np.zeros(D)
    upper = np.ones(D)

    offspring = sbx_crossover_numba(parents, 1.0, 20.0, lower, upper)

    # It's possible for SBX to produce same values if parents are identical or by chance,
    # but with random parents and high prob, full equality is extremely unlikely.
    assert not np.array_equal(offspring, parents)


@pytest.mark.numba
def test_numba_backend_integration():
    """Verify NumbaBackend uses the new ops correctly."""
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    kernel = NumbaKernel()
    N, D = 10, 2
    X = np.random.rand(N, D)
    params = {"prob": 0.5, "eta": 20.0}

    # Mutation
    kernel.polynomial_mutation(X, params, None, 0.0, 1.0)
    assert np.all(X >= 0.0)
    assert np.all(X <= 1.0)

    # Crossover
    parents = np.random.rand(20, D)  # 10 pairs
    offspring = kernel.sbx_crossover(parents, params, None, 0.0, 1.0)
    assert offspring.shape == parents.shape
    assert np.all(offspring >= 0.0)
    assert np.all(offspring <= 1.0)

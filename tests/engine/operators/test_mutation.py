"""Deterministic regression tests for real-valued mutation operators."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.operators.impl.real import PolynomialMutation
from vamos.operators.impl.real import (
    GaussianMutation,
    NonUniformMutation,
    UniformResetMutation,
    VariationWorkspace,
)

POPULATION = np.array(
    [
        [-1.0, 0.0, 1.5],
        [1.0, 2.5, 3.0],
        [-2.5, 1.0, 0.5],
        [2.0, -1.5, 4.0],
        [0.5, -0.5, 2.0],
    ],
    dtype=float,
)

LOWER = np.array([-5.0, -5.0, -5.0])
UPPER = np.array([5.0, 5.0, 5.0])


def test_polynomial_mutation_matches_reference_output():
    operator = PolynomialMutation(prob_mutation=0.7, eta=12.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(7)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-0.9133301035, 0.0, 1.5],
            [1.212905412, 4.6180787172, 3.0],
            [-3.3035316854, 1.0, 0.5],
            [0.1621495702, -1.4767732, 3.9463135163],
            [1.7901325422, -0.2726208852, 2.0217916808],
        ]
    )
    assert_allclose(mutated, expected)


def test_gaussian_mutation_with_bounds_clamping():
    operator = GaussianMutation(
        prob_mutation=0.6,
        sigma=np.array([0.1, 0.2, 0.05]),
        lower=LOWER,
        upper=UPPER,
    )
    rng = np.random.default_rng(8)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-0.9389648854, 0.0, 1.5720008363],
            [1.0, 2.5, 3.0181169297],
            [-2.4741889733, 0.6721104075, 0.5180077616],
            [1.98815023, -1.5479495698, 3.9922349169],
            [0.5218971705, -0.8632791323, 2.0],
        ]
    )
    assert_allclose(mutated, expected)


def test_uniform_reset_mutation_resamples_within_bounds():
    operator = UniformResetMutation(prob_mutation=0.5, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(9)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-1.0, -1.8439651002, 1.5],
            [1.0, 2.5, 3.0],
            [-2.5, 1.0, 4.8623854928],
            [3.82862567, 4.1281005178, 2.082069141],
            [0.5449687726, -0.5, 2.0],
        ]
    )
    assert_allclose(mutated, expected)


def test_non_uniform_mutation_with_workspace_matches_reference_output():
    operator = NonUniformMutation(
        prob_mutation=0.6,
        perturbation=0.8,
        lower=LOWER,
        upper=UPPER,
        workspace=VariationWorkspace(),
    )
    rng = np.random.default_rng(10)
    mutated = operator(POPULATION, rng)
    expected = np.array(
        [
            [-1.0, -5.0, 1.5],
            [5.0, 1.7449649285, -3.9558739707],
            [-2.5, 1.0, 0.2579058442],
            [2.0, -1.5, 5.0],
            [-2.8026855648, -0.5, 2.0],
        ]
    )
    assert_allclose(mutated, expected)

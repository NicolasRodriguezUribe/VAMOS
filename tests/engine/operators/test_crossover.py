"""Deterministic regression tests for real-valued crossover operators."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.operators.real import SBXCrossover
from vamos.operators.real import (
    ArithmeticCrossover,
    BLXAlphaCrossover,
    DifferentialCrossover,
)

PARENTS = np.array(
    [
        [[-1.0, 0.0, 1.5], [1.0, 2.5, 3.0]],
        [[-2.5, 1.0, 0.5], [2.0, -1.5, 4.0]],
        [[0.5, -0.5, 2.0], [-0.5, 0.25, -2.0]],
    ],
    dtype=float,
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


def test_sbx_crossover_matches_reference_output():
    operator = SBXCrossover(prob_crossover=1.0, eta=15.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(2)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[-0.99099557, 0.0, 1.5], [0.99099557, 2.5, 3.0]],
            [[2.04398263, -1.73268703, 4.05049481], [-2.543982, 1.23268703, 0.44941752]],
            [[0.5, -0.5, 1.95449037], [-0.5, 0.25, -1.95449037]],
        ]
    )
    assert_allclose(offspring, expected)


def test_blx_alpha_crossover_with_clipping_matches_reference_output():
    operator = BLXAlphaCrossover(alpha=0.35, prob_crossover=1.0, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(3)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[0.2793509226, -0.4749532705, 2.0794736976], [-0.2358647306, 1.6188939286, 2.8564863576]],
            [[-0.4102575692, -1.6961096128, 3.6457340509], [3.2404444995, -1.1671450541, 3.1338558821]],
            [[-0.6567575661, -0.2636840571, 0.1138332418], [0.3335671943, -0.389281045, -3.3898674321]],
        ]
    )
    assert_allclose(offspring, expected)


def test_arithmetic_crossover():
    operator = ArithmeticCrossover(prob_crossover=1.0)
    rng = np.random.default_rng(4)
    offspring = operator(PARENTS, rng)
    expected = np.array(
        [
            [[0.83832795, 2.29790994, 2.87874596], [-0.83832795, 0.20209006, 1.62125404]],
            [[-0.73310124, 0.01838958, 1.87425459], [0.23310124, -0.51838958, 2.62574541]],
            [[-0.12351342, -0.03236494, -0.49405366], [0.12351342, -0.21763506, 0.49405366]],
        ]
    )
    assert_allclose(offspring, expected)


def test_differential_crossover_respects_bounds_and_matches_reference_output():
    operator = DifferentialCrossover(F=0.6, CR=0.8, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(5)
    trial = operator(POPULATION, rng)
    expected = np.array(
        [
            [-1.9, 0.6, 0.3],
            [1.9, 1.9, 3.0],
            [-4.3, 1.0, 0.5],
            [3.8, -2.4, 4.9],
            [-2.2, 1.0, -0.1],
        ]
    )
    assert_allclose(trial, expected)

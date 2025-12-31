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
            [[0.89955206, -0.0175751345, 3.0291881263], [-0.89955206, 2.5175751331, 1.4708118737]],
            [[-2.3664956707, -1.3391039234, 0.56419362], [1.8664960099, 0.8391039234, 3.935768272]],
            [[0.5119576484, 0.2531300826, 1.8550744027], [-0.5119576484, -0.5031300826, -1.8550744027]],
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
            [[0.8383279522, 0.98161042, 2.4352701234], [-0.8383279522, 1.51838958, 2.0647298766]],
            [[-1.6085554314, -1.0636804596, 0.9492765403], [1.1085554314, 0.5636804596, 3.5507234597]],
            [[0.0439414008, -0.4266613098, -0.0913859046], [-0.0439414008, 0.1766613098, 0.0913859046]],
        ]
    )
    assert_allclose(offspring, expected)


def test_differential_crossover_respects_bounds_and_matches_reference_output():
    operator = DifferentialCrossover(F=0.6, CR=0.8, lower=LOWER, upper=UPPER)
    rng = np.random.default_rng(5)
    trial = operator(POPULATION, rng)
    expected = np.array(
        [
            [-0.1, -2.4, 2.5],
            [2.9, -2.1, 4.6],
            [-2.5, 2.8, 2.7],
            [-1.6, -1.4, 0.5],
            [-4.3, -0.5, -1.0],
        ]
    )
    assert_allclose(trial, expected)

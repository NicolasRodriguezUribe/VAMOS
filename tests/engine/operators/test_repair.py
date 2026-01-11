"""Deterministic regression tests for repair helpers."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.operators.impl.real import ClampRepair, ReflectRepair, ResampleRepair, RoundRepair

VIOLATING = np.array(
    [
        [-2.5, 0.5, 2.5],
        [1.2, -0.4, -3.1],
        [3.0, 4.2, 1.2],
    ],
    dtype=float,
)

LOWER = np.array([-1.5, 0.0, -2.0])
UPPER = np.array([1.5, 3.5, 2.0])


def test_clamp_repair_limits_values_to_bounds():
    operator = ClampRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.5, 0.5, 2.0],
            [1.2, 0.0, -2.0],
            [1.5, 3.5, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_reflect_repair_reflects_out_of_bounds_entries():
    operator = ReflectRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-0.5, 0.5, 1.5],
            [1.2, 0.4, -0.9],
            [0.0, 2.8, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_resample_repair_resamples_violated_genes():
    operator = ResampleRepair()
    rng = np.random.default_rng(11)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.1142893917, 0.5, 0.4059934305],
            [1.2, 0.517741296, 1.7128440918],
            [-1.2887382715, 0.4542088229, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_round_repair_rounds_and_clamps():
    operator = RoundRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.5, 0.0, 2.0],
            [1.0, 0.0, -2.0],
            [1.5, 3.5, 1.0],
        ]
    )
    assert_allclose(repaired, expected)

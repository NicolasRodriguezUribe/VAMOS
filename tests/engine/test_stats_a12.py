"""Tests for Vargha-Delaney A12 effect size."""

from __future__ import annotations

import numpy as np

from vamos.engine.tuning.racing.stats import a12_magnitude, vargha_delaney_a12


def test_a12_identical_samples():
    a = np.array([1.0, 2.0, 3.0])
    assert vargha_delaney_a12(a, a) == 0.5


def test_a12_completely_dominant():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([1.0, 2.0, 3.0])
    assert vargha_delaney_a12(a, b) == 1.0
    assert vargha_delaney_a12(b, a) == 0.0


def test_a12_symmetry():
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 4.0, 6.0])
    assert abs(vargha_delaney_a12(a, b) + vargha_delaney_a12(b, a) - 1.0) < 1e-10


def test_a12_empty_returns_half():
    assert vargha_delaney_a12(np.array([]), np.array([1.0])) == 0.5
    assert vargha_delaney_a12(np.array([1.0]), np.array([])) == 0.5


def test_a12_known_value():
    # a=[3,5], b=[1,2,4]: 3>1,3>2,5>1,5>2,5>4 = 5 wins out of 6 → A12 = 5/6
    a = np.array([3.0, 5.0])
    b = np.array([1.0, 2.0, 4.0])
    expected = 5.0 / 6.0
    assert abs(vargha_delaney_a12(a, b) - expected) < 1e-10


def test_magnitude_negligible():
    assert a12_magnitude(0.5) == "negligible"
    assert a12_magnitude(0.54) == "negligible"


def test_magnitude_small():
    assert a12_magnitude(0.56) == "small"
    assert a12_magnitude(0.63) == "small"


def test_magnitude_medium():
    assert a12_magnitude(0.64) == "medium"
    assert a12_magnitude(0.70) == "medium"


def test_magnitude_large():
    assert a12_magnitude(0.72) == "large"
    assert a12_magnitude(0.95) == "large"


def test_magnitude_symmetric():
    assert a12_magnitude(0.3) == "medium"  # |0.3 - 0.5| = 0.2
    assert a12_magnitude(0.1) == "large"   # |0.1 - 0.5| = 0.4

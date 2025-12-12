import numpy as np
import pytest

import vamos.algorithm.hypervolume as hv


def test_hypervolume_impl_2d_matches_known_value():
    points = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=float)
    ref = np.array([1.0, 1.0], dtype=float)
    assert hv._hypervolume_impl(points, ref) == pytest.approx(0.37)


def test_hypervolume_impl_2d_ignores_dominated_points():
    points = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    ref = np.array([3.0, 3.0], dtype=float)
    assert hv._hypervolume_impl(points, ref) == pytest.approx(4.0)


def test_hypervolume_contributions_2d_matches_bruteforce():
    points = np.array(
        [
            [0.2, 0.8],
            [0.8, 0.2],
            [0.5, 0.9],  # dominated by [0.2, 0.8]
            [0.2, 0.8],  # duplicate of the first point
        ],
        dtype=float,
    )
    ref = np.array([1.0, 1.0], dtype=float)
    hv_full = hv._hypervolume_impl(points, ref)
    expected = np.array(
        [hv_full - hv._hypervolume_impl(np.delete(points, i, axis=0), ref) for i in range(points.shape[0])],
        dtype=float,
    )
    assert hv._hypervolume_contributions_2d(points, ref) == pytest.approx(expected)


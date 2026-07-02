import numpy as np
import pytest

import vamos.foundation.quality_indicators.hypervolume as hv


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


def test_hypervolume_rejects_non_dominating_reference_point_by_default():
    points = np.array([[0.2, 1.2], [0.5, 0.5]], dtype=float)
    ref = np.array([1.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="reference_point must dominate"):
        hv.hypervolume(points, ref)
    with pytest.raises(ValueError, match="reference_point must dominate"):
        hv.hypervolume_contributions(points, ref)

    assert hv.hypervolume(points, ref, allow_ref_expand=True) >= 0.0
    contributions = hv.hypervolume_contributions(points, ref, allow_ref_expand=True)
    assert contributions.shape == (points.shape[0],)


def test_hypervolume_rejects_non_finite_points_and_reference():
    ref = np.array([1.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        hv.hypervolume(np.array([[np.nan, 0.2]], dtype=float), ref)
    with pytest.raises(ValueError, match="finite"):
        hv.hypervolume_contributions(np.array([[0.2, 0.2]], dtype=float), np.array([np.inf, 1.0]))


def test_hypervolume_supports_three_objective_fronts():
    points = np.array([[1.0, 1.0, 1.0]], dtype=float)
    ref = np.array([2.0, 2.0, 2.0], dtype=float)

    assert hv.hypervolume(points, ref) == pytest.approx(1.0)
    assert hv.hypervolume_contributions(points, ref) == pytest.approx(np.array([1.0]))

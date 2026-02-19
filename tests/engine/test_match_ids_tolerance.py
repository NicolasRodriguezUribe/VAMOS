"""Tests for the match_ids helper with floating-point tolerance."""

import numpy as np

from vamos.engine.algorithm.nsgaii.helpers import match_ids


class TestMatchIdsExact:
    """match_ids should work with exact row matches (pre-fix behaviour)."""

    def test_exact_match_single_row(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([100, 200])
        new_X = np.array([[3.0, 4.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 200

    def test_exact_match_multiple_rows(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        combined_ids = np.array([10, 20, 30])
        new_X = np.array([[5.0, 6.0], [1.0, 2.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        np.testing.assert_array_equal(ids, [30, 10])

    def test_exact_match_all_rows(self):
        combined_X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        combined_ids = np.array([7, 8, 9])
        new_X = combined_X.copy()

        ids = match_ids(new_X, combined_X, combined_ids)

        np.testing.assert_array_equal(ids, [7, 8, 9])


class TestMatchIdsFloatDrift:
    """match_ids should tolerate small floating-point drift (<=1e-12)."""

    def test_tolerates_float_drift(self):
        # Classic floating-point issue: 0.1 + 0.2 != 0.3 exactly.
        combined_X = np.array([[0.1 + 0.2, 0.5], [0.7, 0.8]])
        combined_ids = np.array([10, 20])
        new_X = np.array([[0.3, 0.5]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 10

    def test_tolerates_tiny_perturbation(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([42, 99])
        # Add a perturbation well within the 1e-12 tolerance.
        new_X = np.array([[1.0 + 1e-14, 2.0 - 1e-14]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 42

    def test_drift_just_below_tolerance_boundary(self):
        combined_X = np.array([[5.0, 6.0]])
        combined_ids = np.array([55])
        # Just under the 1e-12 tolerance; should still match.
        new_X = np.array([[5.0 + 9e-13, 6.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 55


class TestMatchIdsNoMatch:
    """match_ids should return -1 for rows that have no close match."""

    def test_no_match_returns_negative_one(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([10, 20])
        new_X = np.array([[9.0, 9.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == -1

    def test_drift_beyond_tolerance_returns_negative_one(self):
        combined_X = np.array([[1.0, 2.0]])
        combined_ids = np.array([10])
        # Perturbation larger than the 1e-12 tolerance.
        new_X = np.array([[1.0 + 1e-10, 2.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == -1

    def test_partial_row_match_is_not_enough(self):
        combined_X = np.array([[1.0, 2.0, 3.0]])
        combined_ids = np.array([10])
        # First two columns match but last column differs.
        new_X = np.array([[1.0, 2.0, 999.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == -1

    def test_mixed_matches_and_misses(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([10, 20])
        new_X = np.array([[3.0, 4.0], [7.0, 8.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 20
        assert ids[1] == -1


class TestMatchIdsMultipleCandidates:
    """When multiple rows match, match_ids should return the first match."""

    def test_duplicate_rows_returns_first_id(self):
        combined_X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([10, 20, 30])
        new_X = np.array([[1.0, 2.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 10

    def test_near_duplicate_rows_returns_first_id(self):
        # Two rows that are both within tolerance of the query.
        combined_X = np.array([[1.0, 2.0], [1.0 + 1e-14, 2.0 - 1e-14]])
        combined_ids = np.array([100, 200])
        new_X = np.array([[1.0, 2.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == 100


class TestMatchIdsEmptyArrays:
    """match_ids should handle edge cases with empty arrays gracefully."""

    def test_empty_new_X(self):
        combined_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        combined_ids = np.array([10, 20])
        new_X = np.empty((0, 2))

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids.shape == (0,)

    def test_empty_combined_X(self):
        combined_X = np.empty((0, 2))
        combined_ids = np.empty(0, dtype=int)
        new_X = np.array([[1.0, 2.0]])

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids[0] == -1

    def test_both_empty(self):
        combined_X = np.empty((0, 2))
        combined_ids = np.empty(0, dtype=int)
        new_X = np.empty((0, 2))

        ids = match_ids(new_X, combined_X, combined_ids)

        assert ids.shape == (0,)

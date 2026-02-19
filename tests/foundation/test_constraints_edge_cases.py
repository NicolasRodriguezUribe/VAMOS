"""
Edge-case tests for constraint handling utilities and strategies.

Covers the behaviour of ``compute_violation``, ``is_feasible``,
``compute_constraint_info``, and all four constraint-handling strategies
when the constraint matrix *G* is ``None`` (unconstrained problems) or
when solutions sit exactly on the constraint boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.constraints import (
    ConstraintInfo,
    CVAsObjectiveStrategy,
    EpsilonConstraintStrategy,
    FeasibilityFirstStrategy,
    PenaltyCVStrategy,
    compute_constraint_info,
)
from vamos.foundation.constraints.utils import compute_violation, is_feasible

# ------------------------------------------------------------------ #
#  compute_violation with G=None                                      #
# ------------------------------------------------------------------ #


class TestComputeViolationGNone:
    """compute_violation must gracefully handle unconstrained (G=None) inputs."""

    def test_returns_empty_array_by_default(self):
        # Arrange
        G = None

        # Act
        result = compute_violation(G)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == float

    def test_returns_zeros_of_length_n(self):
        # Arrange
        G = None
        n = 5

        # Act
        result = compute_violation(G, n=n)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)
        np.testing.assert_array_equal(result, np.zeros(n))

    def test_returns_zeros_of_length_one(self):
        # Arrange / Act
        result = compute_violation(None, n=1)

        # Assert
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_n_zero_same_as_default(self):
        # Arrange / Act
        default_result = compute_violation(None)
        explicit_result = compute_violation(None, n=0)

        # Assert
        np.testing.assert_array_equal(default_result, explicit_result)


# ------------------------------------------------------------------ #
#  is_feasible with G=None                                            #
# ------------------------------------------------------------------ #


class TestIsFeasibleGNone:
    """is_feasible must return all-True masks for unconstrained problems."""

    def test_returns_empty_true_array_by_default(self):
        # Arrange
        G = None

        # Act
        result = is_feasible(G)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        assert result.dtype == bool

    def test_returns_all_true_of_length_n(self):
        # Arrange
        G = None
        n = 7

        # Act
        result = is_feasible(G, n=n)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)
        assert result.all()
        assert result.dtype == bool

    def test_returns_all_true_of_length_one(self):
        # Arrange / Act
        result = is_feasible(None, n=1)

        # Assert
        assert result.shape == (1,)
        assert result[0] is np.bool_(True)

    def test_n_zero_same_as_default(self):
        # Arrange / Act
        default_result = is_feasible(None)
        explicit_result = is_feasible(None, n=0)

        # Assert
        np.testing.assert_array_equal(default_result, explicit_result)


# ------------------------------------------------------------------ #
#  is_feasible with eps tolerance (boundary constraints)              #
# ------------------------------------------------------------------ #


class TestIsFeasibleEpsTolerance:
    """is_feasible must honour the *eps* tolerance parameter."""

    def test_exact_zero_is_feasible_with_zero_eps(self):
        # Arrange -- g(x) = 0.0 is on the boundary (satisfied with <= 0)
        G = np.array([[0.0]])

        # Act
        result = is_feasible(G, eps=0.0)

        # Assert
        assert result[0] is np.bool_(True)

    def test_small_positive_infeasible_without_eps(self):
        # Arrange
        G = np.array([[1e-8]])

        # Act
        result = is_feasible(G, eps=0.0)

        # Assert
        assert result[0] is np.bool_(False)

    def test_small_positive_feasible_with_eps(self):
        # Arrange -- violation of 1e-8 is within tolerance 1e-6
        G = np.array([[1e-8]])

        # Act
        result = is_feasible(G, eps=1e-6)

        # Assert
        assert result[0] is np.bool_(True)

    def test_violation_exactly_at_eps_is_feasible(self):
        # Arrange -- g(x) = eps should be treated as satisfied (<=)
        eps = 0.01
        G = np.array([[eps]])

        # Act
        result = is_feasible(G, eps=eps)

        # Assert
        assert result[0] is np.bool_(True)

    def test_violation_slightly_above_eps_is_infeasible(self):
        # Arrange
        eps = 0.01
        G = np.array([[eps + 1e-12]])

        # Act
        result = is_feasible(G, eps=eps)

        # Assert
        assert result[0] is np.bool_(False)

    def test_multiple_constraints_with_eps(self):
        # Arrange -- two constraints: one within eps, one far violated
        G = np.array(
            [
                [0.005, -1.0],  # first within eps=0.01, second satisfied
                [0.005, 0.1],  # first within eps, second violated beyond eps
            ]
        )

        # Act
        result = is_feasible(G, eps=0.01)

        # Assert
        assert result[0] is np.bool_(True)
        assert result[1] is np.bool_(False)


# ------------------------------------------------------------------ #
#  compute_constraint_info with G=None                                #
# ------------------------------------------------------------------ #


class TestComputeConstraintInfoGNone:
    """compute_constraint_info(G=None) must NOT raise; it should return
    a trivially-feasible ConstraintInfo."""

    def test_does_not_raise(self):
        # Arrange / Act / Assert -- no exception expected
        info = compute_constraint_info(None)
        assert isinstance(info, ConstraintInfo)

    def test_returns_empty_g(self):
        # Arrange / Act
        info = compute_constraint_info(None)

        # Assert
        assert info.G.shape == (0, 0)
        assert info.G.dtype == float

    def test_returns_zero_cv(self):
        # Arrange / Act
        info = compute_constraint_info(None)

        # Assert
        assert info.cv.shape == (0,)
        np.testing.assert_array_equal(info.cv, np.zeros(0))

    def test_returns_empty_feasible_mask(self):
        # Arrange / Act
        info = compute_constraint_info(None)

        # Assert
        assert info.feasible_mask.shape == (0,)
        assert info.feasible_mask.dtype == bool

    def test_eps_parameter_has_no_effect_when_g_none(self):
        # Arrange / Act
        info_default = compute_constraint_info(None, eps=0.0)
        info_large_eps = compute_constraint_info(None, eps=100.0)

        # Assert
        np.testing.assert_array_equal(info_default.cv, info_large_eps.cv)
        np.testing.assert_array_equal(info_default.feasible_mask, info_large_eps.feasible_mask)


# ------------------------------------------------------------------ #
#  All four strategies with G=None                                    #
# ------------------------------------------------------------------ #


class TestStrategiesWithGNone:
    """Every strategy.rank(F, G=None) must return only the aggregated
    objectives, unmodified by any constraint penalty."""

    @pytest.fixture()
    def F(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 2.0],
                [0.5, 0.3],
                [3.0, 4.0],
            ]
        )

    def test_feasibility_first_g_none(self, F: np.ndarray):
        # Arrange
        strategy = FeasibilityFirstStrategy(objective_aggregator="sum")
        expected = np.sum(F, axis=1)

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores, expected)

    def test_penalty_cv_g_none(self, F: np.ndarray):
        # Arrange
        strategy = PenaltyCVStrategy(penalty_lambda=1e6, objective_aggregator="sum")
        expected = np.sum(F, axis=1)

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores, expected)

    def test_cv_as_objective_g_none(self, F: np.ndarray):
        # Arrange
        strategy = CVAsObjectiveStrategy(objective_aggregator="sum")
        expected = np.sum(F, axis=1)

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores, expected)

    def test_epsilon_constraint_g_none(self, F: np.ndarray):
        # Arrange
        strategy = EpsilonConstraintStrategy(epsilon=0.5, objective_aggregator="sum")
        expected = np.sum(F, axis=1)

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores, expected)

    def test_all_strategies_consistent_with_g_none(self, F: np.ndarray):
        """All strategies must produce the same ranking when unconstrained."""
        # Arrange
        strategies = [
            FeasibilityFirstStrategy(objective_aggregator="sum"),
            PenaltyCVStrategy(penalty_lambda=500.0, objective_aggregator="sum"),
            CVAsObjectiveStrategy(objective_aggregator="sum"),
            EpsilonConstraintStrategy(epsilon=0.0, objective_aggregator="sum"),
        ]

        # Act
        rankings = [np.argsort(s.rank(F, G=None)) for s in strategies]

        # Assert -- all orderings must be identical
        for ranking in rankings[1:]:
            np.testing.assert_array_equal(rankings[0], ranking)

    def test_max_aggregator_g_none(self, F: np.ndarray):
        """Ensure the 'max' aggregator works with G=None as well."""
        # Arrange
        strategy = FeasibilityFirstStrategy(objective_aggregator="max")
        expected = np.max(F, axis=1)

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores, expected)

    def test_none_aggregator_g_none(self, F: np.ndarray):
        """The 'none' aggregator should return zeros when unconstrained."""
        # Arrange
        strategy = PenaltyCVStrategy(penalty_lambda=1.0, objective_aggregator="none")

        # Act
        scores = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_equal(scores, np.zeros(F.shape[0]))


# ------------------------------------------------------------------ #
#  Constraint boundary: solutions exactly at g(x) = 0                 #
# ------------------------------------------------------------------ #


class TestConstraintBoundary:
    """Behaviour when all constraint values are exactly 0.0 (the boundary)."""

    @pytest.fixture()
    def G_boundary(self) -> np.ndarray:
        """All solutions sit exactly on the constraint boundary."""
        return np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

    def test_compute_violation_boundary_is_zero(self, G_boundary: np.ndarray):
        # Arrange / Act
        cv = compute_violation(G_boundary)

        # Assert
        np.testing.assert_array_equal(cv, np.zeros(3))

    def test_is_feasible_boundary(self, G_boundary: np.ndarray):
        # Arrange / Act
        feas = is_feasible(G_boundary)

        # Assert -- g(x) <= 0 is satisfied when g(x) = 0
        assert feas.all()

    def test_constraint_info_boundary(self, G_boundary: np.ndarray):
        # Arrange / Act
        info = compute_constraint_info(G_boundary)

        # Assert
        np.testing.assert_array_equal(info.cv, np.zeros(3))
        assert info.feasible_mask.all()

    def test_mixed_boundary_and_violated(self):
        # Arrange -- first solution on boundary, second slightly violated
        G = np.array(
            [
                [0.0, 0.0],
                [0.0, 1e-15],
            ]
        )

        # Act
        feas = is_feasible(G)
        cv = compute_violation(G)

        # Assert
        assert feas[0] is np.bool_(True)
        assert feas[1] is np.bool_(False)
        assert cv[0] == 0.0
        assert cv[1] > 0.0

    def test_boundary_with_negative_constraints(self):
        # Arrange -- all satisfied: some negative, some zero
        G = np.array(
            [
                [-1.0, 0.0],
                [0.0, -0.5],
                [-2.0, -3.0],
            ]
        )

        # Act
        cv = compute_violation(G)
        feas = is_feasible(G)

        # Assert
        np.testing.assert_array_equal(cv, np.zeros(3))
        assert feas.all()

    def test_feasibility_first_boundary_uses_objective(self):
        """When all solutions are on the boundary (feasible), ranking
        should be determined purely by objective aggregation."""
        # Arrange
        F = np.array([[3.0], [1.0], [2.0]])
        G = np.array([[0.0], [0.0], [0.0]])
        strategy = FeasibilityFirstStrategy(objective_aggregator="sum")

        # Act
        scores = strategy.rank(F, G)
        order = np.argsort(scores)

        # Assert -- ranked by objective: 1.0 < 2.0 < 3.0
        np.testing.assert_array_equal(order, [1, 2, 0])

    def test_penalty_cv_boundary_matches_unconstrained(self):
        """On the boundary, PenaltyCVStrategy should produce the same
        scores as the unconstrained (G=None) case."""
        # Arrange
        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        G_boundary = np.array([[0.0], [0.0]])
        strategy = PenaltyCVStrategy(penalty_lambda=1e6, objective_aggregator="sum")

        # Act
        scores_boundary = strategy.rank(F, G_boundary)
        scores_none = strategy.rank(F, G=None)

        # Assert
        np.testing.assert_array_almost_equal(scores_boundary, scores_none)

    def test_epsilon_strategy_boundary_within_epsilon(self):
        """Slightly positive g(x) values should be feasible if within
        the epsilon tolerance."""
        # Arrange
        F = np.array([[1.0], [2.0], [3.0]])
        G = np.array([[0.05], [0.1], [0.15]])
        strategy = EpsilonConstraintStrategy(epsilon=0.1, objective_aggregator="sum")

        # Act
        scores = strategy.rank(F, G)
        info = compute_constraint_info(G, eps=0.1)

        # Assert -- first two within epsilon, third is not
        assert info.feasible_mask[0] is np.bool_(True)
        assert info.feasible_mask[1] is np.bool_(True)
        assert info.feasible_mask[2] is np.bool_(False)
        # Feasible solutions ranked by objective; infeasible penalised
        assert scores[0] < scores[1]
        assert scores[2] > scores[1]

import numpy as np

from vamos.ux.analysis.mcdm import (
    knee_point_scores,
    reference_point_scores,
    tchebycheff_scores,
    weighted_sum_scores,
)


def test_weighted_sum_scores_basic():
    # Arrange
    F = np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.5]])
    weights = np.array([0.5, 0.5])

    # Act
    result = weighted_sum_scores(F, weights)

    # Assert
    expected = np.array([1.5, 1.0, 1.25])
    assert np.allclose(result.scores, expected)
    assert result.best_index == 1
    assert np.allclose(result.best_point, F[1])


def test_tchebycheff_scores_prefers_reference_proximity():
    # Arrange
    F = np.array([[1.0, 1.0], [0.1, 0.9], [0.8, 0.2]])
    weights = np.array([1.0, 1.0])
    reference = np.array([0.0, 0.0])

    # Act
    result = tchebycheff_scores(F, weights, reference=reference)

    # Assert
    assert result.best_index == 2  # closest to reference under Chebyshev
    assert np.isclose(result.scores[result.best_index], np.min(result.scores))


def test_reference_point_scores_identifies_exact_match():
    # Arrange
    F = np.array([[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]])
    reference = np.array([0.2, 0.2])

    # Act
    result = reference_point_scores(F, reference)

    # Assert
    assert result.best_index == 1
    assert result.scores[1] == 0.0


def test_knee_point_scores_2d():
    # Arrange
    F = np.array([[0.0, 1.0], [0.2, 0.7], [0.4, 0.5], [0.6, 0.4], [1.0, 0.0]])

    # Act
    result = knee_point_scores(F)

    # Assert
    assert result.best_index in {1, 2, 3}
    assert result.scores[result.best_index] == np.min(result.scores)

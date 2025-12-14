import pytest
import numpy as np
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vamos.ux.analysis.stats import (
    FriedmanResult,
    compute_ranks,
    friedman_test,
    pairwise_wilcoxon,
    plot_critical_distance,
)


def test_compute_ranks_higher_and_lower():
    # Arrange
    scores = np.array([[1.0, 2.0, 3.0], [0.5, 0.2, 0.1]])

    # Act
    ranks_high = compute_ranks(scores, higher_is_better=True)
    ranks_low = compute_ranks(scores, higher_is_better=False)

    # Assert
    assert ranks_high[0].tolist() == [3.0, 2.0, 1.0]
    assert ranks_low[0].tolist() == [1.0, 2.0, 3.0]


def test_friedman_test_detects_difference():
    # Arrange
    scores = np.array([[0.9, 0.5, 0.4], [0.8, 0.4, 0.3], [0.85, 0.45, 0.35]])

    # Act
    result = friedman_test(scores, higher_is_better=True)

    # Assert
    assert isinstance(result, FriedmanResult)
    assert result.statistic > 0
    assert result.p_value < 0.1
    assert result.avg_ranks[0] < result.avg_ranks[1]


def test_pairwise_wilcoxon_pair_count_and_order():
    # Arrange
    scores = np.array([[1.0, 0.5, 0.6], [1.1, 0.4, 0.65], [0.9, 0.45, 0.7]])
    algo_names = ["A", "B", "C"]

    # Act
    results = pairwise_wilcoxon(scores, algo_names, higher_is_better=True)

    # Assert
    assert len(results) == 3
    p_ab = next(res.p_value for res in results if res.algo_i == "A" and res.algo_j == "B")
    p_bc = next(res.p_value for res in results if res.algo_i == "B" and res.algo_j == "C")
    assert p_ab <= p_bc


def test_plot_critical_distance_smoke():
    # Arrange
    avg_ranks = np.array([1.2, 2.5, 2.3])
    algo_names = ["A", "B", "C"]

    # Act
    ax = plot_critical_distance(avg_ranks, algo_names, n_problems=10, show=False)

    # Assert
    assert hasattr(ax, "scatter")

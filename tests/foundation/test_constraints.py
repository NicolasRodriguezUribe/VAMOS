import numpy as np
import pytest

from vamos.foundation.constraints import (
    ConstraintInfo,
    CVAsObjectiveStrategy,
    EpsilonConstraintStrategy,
    FeasibilityFirstStrategy,
    PenaltyCVStrategy,
    compute_constraint_info,
    get_constraint_strategy,
)


def test_compute_constraint_info_basic():
    # Arrange
    G = np.array([[-1.0, 0.0], [0.2, -0.1], [0.5, 0.5]])

    # Act
    info = compute_constraint_info(G)

    # Assert
    assert np.allclose(info.cv, [0.0, 0.2, 1.0])
    assert info.feasible_mask.tolist() == [True, False, False]


def test_feasibility_first_ranks_feasible_before_infeasible():
    # Arrange
    F = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, 0.5]])
    G = np.array([[-0.1], [0.2], [-0.2]])
    strategy = FeasibilityFirstStrategy(objective_aggregator="sum")

    # Act
    scores = strategy.rank(F, G)
    order = np.argsort(scores)

    # Assert
    assert order[0] in {0, 2}  # feasible solutions first
    assert order[-1] == 1  # infeasible last


def test_penalty_cv_prefers_feasible_over_better_objective_infeasible():
    # Arrange
    F = np.array([[1.0, 1.0], [0.1, 0.1]])
    G = np.array([[0.0], [0.5]])  # second is infeasible
    strategy = PenaltyCVStrategy(penalty_lambda=10.0, objective_aggregator="sum")

    # Act
    scores = strategy.rank(F, G)

    # Assert
    assert scores[0] < scores[1]


def test_cv_as_objective_prioritizes_lower_violation():
    # Arrange
    F = np.array([[1.0, 2.0], [0.5, 0.5]])
    G = np.array([[0.0], [0.3]])  # second has violation
    strategy = CVAsObjectiveStrategy(objective_aggregator="sum", eps=1e-3)

    # Act
    scores = strategy.rank(F, G)

    # Assert
    assert scores[0] < scores[1]


def test_epsilon_constraint_treats_small_cv_as_feasible():
    # Arrange
    F = np.array([[1.0], [0.2], [0.3]])
    G = np.array([[0.05], [0.0], [0.2]])
    strategy = EpsilonConstraintStrategy(epsilon=0.1, objective_aggregator="sum")

    # Act
    scores = strategy.rank(F, G)
    order = np.argsort(scores)

    # Assert
    assert order[0] in {0, 1}  # two within epsilon
    assert order[-1] == 2  # outside epsilon


def test_factory_creates_strategies_and_raises_on_unknown():
    # Arrange / Act
    strategies = {
        "feasibility_first": get_constraint_strategy("feasibility_first"),
        "penalty_cv": get_constraint_strategy("penalty_cv"),
        "cv_as_objective": get_constraint_strategy("cv_as_objective"),
        "epsilon": get_constraint_strategy("epsilon"),
    }

    # Assert
    assert isinstance(strategies["feasibility_first"], FeasibilityFirstStrategy)
    assert isinstance(strategies["penalty_cv"], PenaltyCVStrategy)
    assert isinstance(strategies["cv_as_objective"], CVAsObjectiveStrategy)
    assert isinstance(strategies["epsilon"], EpsilonConstraintStrategy)
    with pytest.raises(ValueError):
        get_constraint_strategy("unknown")

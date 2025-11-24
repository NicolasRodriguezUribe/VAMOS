import numpy as np

from vamos.objective_reduction import (
    ObjectiveReductionConfig,
    ObjectiveReducer,
    reduce_objectives,
)


def test_constant_objective_removed_unless_mandatory():
    # Arrange
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(20, 2))
    constant = np.zeros((20, 1))
    F = np.hstack([noise, constant])
    reducer = ObjectiveReducer(ObjectiveReductionConfig(method="correlation"))

    # Act
    reducer.fit(F)

    # Assert
    assert 2 not in reducer.selected_indices_
    assert 2 in reducer.removed_indices_

    # Arrange (mandatory keeps constant)
    reducer_keep = ObjectiveReducer(
        ObjectiveReductionConfig(method="correlation", keep_mandatory=(2,))
    )

    # Act
    reducer_keep.fit(F)

    # Assert
    assert 2 in reducer_keep.selected_indices_


def test_highly_correlated_objectives_removed_and_mandatory_preserved():
    # Arrange
    rng = np.random.default_rng(1)
    base = rng.normal(size=30)
    obj0 = base
    obj1 = base + rng.normal(scale=0.001, size=30)  # highly correlated
    obj2 = rng.normal(size=30)
    F = np.vstack([obj0, obj1, obj2]).T
    reducer = ObjectiveReducer(ObjectiveReductionConfig(method="correlation", corr_threshold=0.95))

    # Act
    reducer.fit(F)

    # Assert
    selected = set(reducer.selected_indices_)
    assert 2 in selected
    assert (0 in selected) ^ (1 in selected)

    # Arrange (mandate obj1)
    reducer_mand = ObjectiveReducer(
        ObjectiveReductionConfig(method="correlation", corr_threshold=0.95, keep_mandatory=(1,))
    )

    # Act
    reducer_mand.fit(F)

    # Assert
    assert 1 in reducer_mand.selected_indices_


def test_angle_based_diversity_prefers_opposite_trends():
    # Arrange
    x = np.linspace(0, 1, 50)
    obj0 = x
    obj1 = -x
    obj2 = x + 0.001  # near duplicate of obj0
    F = np.vstack([obj0, obj1, obj2]).T
    reducer = ObjectiveReducer(
        ObjectiveReductionConfig(method="angle", angle_threshold_deg=10.0, target_dim=2)
    )

    # Act
    reducer.fit(F)

    # Assert
    selected = set(reducer.selected_indices_)
    assert {0, 1} <= selected
    assert 2 not in selected


def test_target_dim_respected_for_both_methods():
    # Arrange
    rng = np.random.default_rng(2)
    F = rng.normal(size=(40, 5))

    # Act
    corr_reducer = ObjectiveReducer(
        ObjectiveReductionConfig(method="correlation", target_dim=2)
    ).fit(F)
    angle_reducer = ObjectiveReducer(
        ObjectiveReductionConfig(method="angle", target_dim=2)
    ).fit(F)

    # Assert
    assert corr_reducer.selected_indices_.size == 2
    assert angle_reducer.selected_indices_.size == 2


def test_keep_mandatory_prevents_removal():
    # Arrange
    rng = np.random.default_rng(3)
    base = rng.normal(size=25)
    obj0 = base
    obj1 = base + 0.0001  # highly correlated with obj0
    F = np.vstack([obj0, obj1]).T

    # Act
    reducer = ObjectiveReducer(
        ObjectiveReductionConfig(method="correlation", corr_threshold=0.9, keep_mandatory=(1,))
    ).fit(F)
    angle_reducer = ObjectiveReducer(
        ObjectiveReductionConfig(method="angle", keep_mandatory=(1,))
    ).fit(F)

    # Assert
    assert 1 in reducer.selected_indices_
    assert 1 in angle_reducer.selected_indices_


def test_reduce_objectives_wrapper_returns_expected_shape_and_sorted_indices():
    # Arrange
    rng = np.random.default_rng(4)
    F = rng.normal(size=(10, 3))

    # Act
    reduced, indices = reduce_objectives(F, method="correlation", target_dim=2)

    # Assert
    assert reduced.shape[0] == F.shape[0]
    assert reduced.shape[1] == indices.size
    assert np.all(np.diff(indices) >= 0)

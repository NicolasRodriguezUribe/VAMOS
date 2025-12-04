import numpy as np

from vamos.analysis.objective_reduction import reduce_objectives, ObjectiveReducer


def test_correlation_reduction_removes_duplicates():
    # obj2 is a duplicate of obj1; obj3 is unique
    x = np.linspace(0, 1, 20)
    F = np.stack([x, x + 0.0, 1 - x], axis=1)
    F_red, selected = reduce_objectives(F, target_dim=2, method="correlation")
    assert F_red.shape[1] == 2
    # Expect to keep objective 0 and 2 (or 1 and 2), not both 0 and 1
    assert 2 in selected
    assert len(set(selected)) == 2


def test_angle_reduction_prefers_diverse_directions():
    # Three orthogonal-ish directions
    a = np.linspace(0, 1, 30)
    b = np.linspace(0, 2, 30)
    c = np.sin(np.linspace(0, np.pi, 30))
    F = np.stack([a, b, c], axis=1)
    reducer = ObjectiveReducer(method="angle")
    selected = reducer.reduce(F, target_dim=2)
    assert len(selected) == 2
    # Should not pick two identical objectives
    assert len(set(selected)) == 2


def test_mandatory_keep_is_respected():
    F = np.random.rand(10, 4)
    mandatory = [1, 3]
    selected = ObjectiveReducer().reduce(F, target_dim=2, mandatory_keep=mandatory)
    assert all(idx in selected for idx in mandatory)


def test_hybrid_respects_thresholds_and_keeps_diversity():
    x = np.linspace(0, 1, 25)
    F = np.stack([x, x + 1e-6, 1 - x, np.cos(x)], axis=1)
    # With target_dim=2 should keep one of first pair and something diverse
    reducer = ObjectiveReducer(method="hybrid")
    selected = reducer.reduce(F, target_dim=2)
    assert len(set(selected)) == 2
    assert 2 in selected or 3 in selected

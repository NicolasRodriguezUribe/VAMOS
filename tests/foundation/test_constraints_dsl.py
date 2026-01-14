import numpy as np
import pytest

from vamos.foundation.constraints.dsl import build_constraint_evaluator, constraint_model


def test_constraint_evaluator_matches_manual():
    with constraint_model(n_vars=2) as cm:
        x1, x2 = cm.vars("x1", "x2")
        cm.add(x1 + x2 <= 1)
        cm.add(x1 * x1 + x2 >= 0)
    evaluator = build_constraint_evaluator(cm)
    X = np.array([[0.5, 0.4], [0.8, 0.5]])
    violations = evaluator(X)
    # First point feasible for both
    assert np.allclose(violations[0], [0.0, 0.0])
    # Second point violates first constraint: 0.8+0.5-1 = 0.3
    assert violations[1, 0] > 0


def test_constraint_scalar_accepts_0d_array():
    with constraint_model(n_vars=1) as cm:
        (x0,) = cm.vars("x0")
        cm.add(x0 <= np.array(1.0))
    evaluator = build_constraint_evaluator(cm)
    X = np.array([[0.5], [1.5]])
    violations = evaluator(X)
    assert np.allclose(violations[:, 0], [0.0, 0.5])


def test_constraint_vector_constants_rejected():
    with constraint_model(n_vars=1) as cm:
        (x0,) = cm.vars("x0")
        with pytest.raises(TypeError, match="Vector constants are not supported"):
            cm.add(x0 <= np.array([1.0, 2.0]))

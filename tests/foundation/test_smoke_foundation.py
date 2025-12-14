import numpy as np
import pytest

from vamos.foundation.problem.zdt1 import ZDT1Problem


@pytest.mark.smoke
def test_foundation_problem_and_kernel_eval():
    problem = ZDT1Problem(n_var=4)
    X = np.random.rand(5, problem.n_var)
    F = np.zeros((X.shape[0], problem.n_obj))
    problem.evaluate(X, {"F": F})
    assert F.shape == (5, problem.n_obj)
    assert np.isfinite(F).all()

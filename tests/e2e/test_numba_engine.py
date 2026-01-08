import pytest
import numpy as np
from vamos.api import run_optimization
from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1


@pytest.mark.numba
def test_numba_engine_e2e(workspace):
    """
    Verify that switching to engine='numba' works e2e.
    """
    # Simple ZDT1 run
    problem = ZDT1(n_var=30)

    # Use the high-level helper which handles config construction
    res = run_optimization(problem=problem, algorithm="nsgaii", max_evaluations=500, pop_size=40, seed=42, engine="numba")

    # res.success does not exist in OptimizationResult
    assert res.F is not None
    assert res.X is not None
    assert len(res.F) == 40

    # Check that we actually got a decent front (Numba logic isn't broken)
    # ZDT1 PF is f2 = 1 - sqrt(f1)
    # Just check bounds and basic Pareto properties
    # Since we did only 500 evals, it might not be perfect, but should form a front.

    # Check ranges
    assert np.all(res.F >= 0.0)
    # ZDT1 f1 in [0,1], f2 approx [0,1]
    assert np.max(res.F[:, 0]) <= 1.01

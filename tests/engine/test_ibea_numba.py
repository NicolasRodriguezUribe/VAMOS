import numpy as np

from vamos.engine.algorithm.config import IBEAConfig
from vamos.engine.algorithm.ibea import IBEA
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_ibea_numba_same_seed_reproducible():
    cfg = (
        IBEAConfig.builder()
        .pop_size(30)
        .selection("tournament", size=2)
        .indicator("epsilon")
        .kappa(0.05)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .build()
    ).to_dict()
    problem = ZDT1Problem(n_var=8)

    result_a = IBEA(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=13)
    result_b = IBEA(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=13)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])

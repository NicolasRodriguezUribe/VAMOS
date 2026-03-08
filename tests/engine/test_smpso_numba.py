import numpy as np

from vamos.engine.algorithm.config import SMPSOConfig
from vamos.engine.algorithm.smpso import SMPSO
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_smpso_numba_same_seed_reproducible():
    cfg = (SMPSOConfig.builder().pop_size(30).archive_size(30).mutation("polynomial", prob=0.1, eta=20.0).build()).to_dict()
    problem = ZDT1Problem(n_var=8)

    result_a = SMPSO(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=17)
    result_b = SMPSO(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=17)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])

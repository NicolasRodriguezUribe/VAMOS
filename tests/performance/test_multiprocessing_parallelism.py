from __future__ import annotations

import os
import time

import numpy as np
import pytest

from vamos.foundation.eval.backends import MultiprocessingEvalBackend


class _PidProblem:
    def __init__(self, *, delay: float = 0.02) -> None:
        self.n_var = 2
        self.n_obj = 1
        self.n_constr = 0
        self.delay = float(delay)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        pid = float(os.getpid())
        if self.delay > 0.0:
            time.sleep(self.delay * float(X.shape[0]))
        out["F"][:, 0] = pid


@pytest.mark.slow
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="Multiprocessing requires >=2 CPUs")
def test_multiprocessing_backend_uses_multiple_processes() -> None:
    problem = _PidProblem(delay=0.02)
    X = np.zeros((12, problem.n_var), dtype=float)
    backend = MultiprocessingEvalBackend(n_workers=3, chunk_size=1)

    result = backend.evaluate(X, problem)
    pids = np.unique(result.F[:, 0])

    assert len(pids) >= 2

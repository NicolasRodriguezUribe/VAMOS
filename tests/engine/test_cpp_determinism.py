from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.foundation.kernel.registry import resolve_kernel


@pytest.mark.parametrize("algorithm", ["nsgaii", "smsemoa", "spea2", "ibea"])
def test_cpp_backend_is_deterministic_for_fixed_seed(algorithm: str) -> None:
    pytest.importorskip("vamospp")
    try:
        resolve_kernel("cpp")
    except ImportError:
        pytest.skip("cpp kernel is not available")

    kwargs = {
        "problem": "zdt1",
        "algorithm": algorithm,
        "engine": "cpp",
        "max_evaluations": 600,
        "pop_size": 30,
        "seed": 42,
    }
    result_a = optimize(**kwargs)
    result_b = optimize(**kwargs)

    assert np.array_equal(np.asarray(result_a.X), np.asarray(result_b.X))
    assert np.array_equal(np.asarray(result_a.F), np.asarray(result_b.F))

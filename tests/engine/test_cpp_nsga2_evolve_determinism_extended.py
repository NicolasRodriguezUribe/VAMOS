from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.foundation.kernel.registry import resolve_kernel


def _cpp_available() -> bool:
    try:
        resolve_kernel("cpp")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _cpp_available(), reason="cpp kernel is not available")
def test_cpp_nsga2_evolve_determinism_extended(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAMOS_ENABLE_CPP_EVOLVE_FASTPATH", "1")
    monkeypatch.setenv("VAMOS_ENABLE_CPP_NATIVE_EVOLVE", "1")

    kwargs = {
        "problem": "zdt1",
        "algorithm": "nsgaii",
        "engine": "cpp",
        "max_evaluations": 3600,
        "pop_size": 120,
        "seed": 2025,
    }

    result_a = optimize(**kwargs)
    result_b = optimize(**kwargs)

    assert np.array_equal(np.asarray(result_a.X), np.asarray(result_b.X))
    assert np.array_equal(np.asarray(result_a.F), np.asarray(result_b.F))

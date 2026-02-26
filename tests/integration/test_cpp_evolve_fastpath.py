from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vamos.foundation.kernel.registry import resolve_kernel


def _cpp_is_available() -> bool:
    try:
        resolve_kernel("cpp")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _cpp_is_available(), reason="cpp kernel is not available")
def test_cpp_nsga2_evolve_fastpath_subprocess_is_stable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        "import numpy as np\n"
        "from vamos import optimize\n"
        "kw = dict(problem='zdt1', algorithm='nsgaii', engine='cpp', max_evaluations=600, pop_size=30, seed=21)\n"
        "a = optimize(**kw)\n"
        "b = optimize(**kw)\n"
        "assert np.array_equal(np.asarray(a.F), np.asarray(b.F))\n"
        "assert np.array_equal(np.asarray(a.X), np.asarray(b.X))\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env["VAMOS_ENABLE_CPP_EVOLVE_FASTPATH"] = "1"
    env["VAMOS_ENABLE_CPP_NATIVE_EVOLVE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert "ok" in proc.stdout

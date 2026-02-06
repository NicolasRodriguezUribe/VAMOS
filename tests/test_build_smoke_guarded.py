from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_build_smoke_guarded() -> None:
    if os.getenv("VAMOS_RUN_BUILD_SMOKE") != "1":
        pytest.skip("Set VAMOS_RUN_BUILD_SMOKE=1 to run build smoke verification.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify_build_smoke.py"
    assert script_path.is_file()

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Build smoke script failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

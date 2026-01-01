from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_smoke() -> None:
    cmd = [sys.executable, "-m", "build"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    raise AssertionError(f"python -m build failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PY_TYPED = REPO_ROOT / "src" / "vamos" / "py.typed"


def _build_wheel(outdir: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(outdir),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        raise AssertionError(f"python -m build --wheel failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")
    wheels = sorted(outdir.glob("*.whl"))
    if not wheels:
        raise AssertionError(f"No wheel produced in {outdir}")
    return wheels[0]


def test_py_typed_present() -> None:
    if not SRC_PY_TYPED.exists():
        raise AssertionError(f"Missing py.typed marker: {SRC_PY_TYPED}")

    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = _build_wheel(Path(tmpdir))
        with zipfile.ZipFile(wheel_path) as wheel:
            names = set(wheel.namelist())
        if "vamos/py.typed" not in names:
            raise AssertionError("Wheel missing vamos/py.typed marker.")

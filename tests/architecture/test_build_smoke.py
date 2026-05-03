from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_build_outputs() -> None:
    for path in [REPO_ROOT / "build", REPO_ROOT / "dist", *(REPO_ROOT / "src").glob("*.egg-info")]:
        if path.exists():
            shutil.rmtree(path)


def _assert_wheel_matches_source() -> None:
    wheels = sorted((REPO_ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise AssertionError("Build completed without producing a wheel under dist/*.whl.")
    wheel_path = wheels[-1]
    src_root = REPO_ROOT / "src"
    unexpected: list[str] = []
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if not name.startswith("vamos/") or not name.endswith(".py"):
                continue
            src_path = src_root / Path(name)
            if not src_path.is_file():
                unexpected.append(name)
    if unexpected:
        preview = "\n".join(f"- {name}" for name in unexpected[:20])
        if len(unexpected) > 20:
            preview += f"\n- ... and {len(unexpected) - 20} more"
        raise AssertionError(f"Wheel contains Python modules not present in src/vamos:\n{preview}")


def test_build_smoke() -> None:
    _clean_build_outputs()
    cmd = [sys.executable, "-m", "build"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        _assert_wheel_matches_source()
        return

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    raise AssertionError(f"python -m build failed.\nExit code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n")

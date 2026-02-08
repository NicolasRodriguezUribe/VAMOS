from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_step(name: str, cmd: list[str], *, cwd: Path) -> int:
    print(f"[STEP] {name}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} (exit code {result.returncode})", file=sys.stderr)
    print("-" * 80)
    return int(result.returncode)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print(f"[INFO] Repository root: {repo_root}")

    build_cmd = [sys.executable, "-m", "build"]
    build_rc = _run_step("Build wheel/sdist", build_cmd, cwd=repo_root)
    if build_rc != 0:
        message = "Build failed."
        combined = ""
        try:
            probe = subprocess.run(build_cmd, cwd=str(repo_root), text=True, capture_output=True, check=False)
            combined = f"{probe.stdout}\n{probe.stderr}".lower()
        except Exception:
            combined = ""
        if "no module named build" in combined:
            message = "Build module missing. Install it first: pip install build"
        print(f"[ERROR] {message}", file=sys.stderr)
        return build_rc

    wheel_files = sorted((repo_root / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheel_files:
        print("[ERROR] No wheel found under dist/*.whl after build.", file=sys.stderr)
        return 1
    wheel_path = wheel_files[-1]
    print(f"[INFO] Using wheel: {wheel_path}")

    with tempfile.TemporaryDirectory(prefix="vamos_build_smoke_") as tmp_dir:
        temp_root = Path(tmp_dir)
        venv_dir = temp_root / "venv"

        rc = _run_step("Create temporary venv", [sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root)
        if rc != 0:
            return rc

        python_bin = _venv_python(venv_dir)
        if not python_bin.exists():
            print(f"[ERROR] Virtual environment python not found: {python_bin}", file=sys.stderr)
            return 1

        rc = _run_step("Install wheel in venv", [str(python_bin), "-m", "pip", "install", str(wheel_path)], cwd=repo_root)
        if rc != 0:
            return rc

        rc = _run_step("Import smoke check", [str(python_bin), "-c", "import vamos; print('ok')"], cwd=repo_root)
        if rc != 0:
            return rc

        rc = _run_step(
            "Assist doctor JSON smoke",
            [str(python_bin), "-m", "vamos.experiment.cli.main", "assist", "doctor", "--json"],
            cwd=repo_root,
        )
        if rc != 0:
            return rc

    print("[PASS] Build smoke verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

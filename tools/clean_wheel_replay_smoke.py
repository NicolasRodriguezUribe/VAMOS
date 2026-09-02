"""Exercise canonical verification and replay using only an installed wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import numpy as np

import vamos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.wheel is not None:
        _run_in_clean_environment(args.wheel.resolve())
        return
    _run_installed_smoke()


def _run_installed_smoke() -> None:
    package_path = Path(vamos.__file__).resolve()
    if "site-packages" not in {part.lower() for part in package_path.parts}:
        raise AssertionError(f"Expected an installed wheel, got {package_path}")
    with tempfile.TemporaryDirectory(prefix="vamos-wheel-smoke-") as raw_root:
        root = Path(raw_root)
        source = root / "source"
        cli_replay = root / "replay-cli"
        python_replay = root / "replay-python"
        moved_replay = root / "replay-moved"
        result = vamos.optimize(
            "zdt1",
            algorithm="nsgaii",
            pop_size=6,
            max_evaluations=12,
            engine="numpy",
            seed=0,
            n_var=6,
        )
        vamos.save_result(result, source)
        before = _snapshot(source)
        inspect_payload = _cli("results", "inspect", str(source), "--json")
        verify_payload = _cli("results", "verify", str(source), "--require-level", "exact", "--json")
        verification = vamos.verify_run(source, require_level="exact")
        cli_payload = _cli("reproduce", str(source), "--output", str(cli_replay), "--json")
        python_report = vamos.reproduce(source, output=python_replay)
        _assert_exact(source, cli_replay)
        _assert_exact(source, python_replay)
        vamos.load_run(cli_replay, verify="all")
        shutil.move(cli_replay, moved_replay)
        moved = vamos.load_run(moved_replay, verify="all")
        if _snapshot(source) != before:
            raise AssertionError("Source run changed during clean-wheel replay smoke.")
        if not (
            inspect_payload["document_type"] == "vamos.run-inspection"
            and verify_payload["effective_replayability"] == "exact"
            and verification.effective_replayability == "exact"
            and verification.optimization_executed is False
            and cli_payload["exact"] is True
            and python_report.exact
            and moved.manifest["lineage"]["comparison"]["status"] == "exact_match"
        ):
            raise AssertionError("Clean-wheel replay evidence is incomplete.")
    print(json.dumps({"status": "passed", "vamos_file": str(package_path)}, sort_keys=True))


def _run_in_clean_environment(wheel: Path) -> None:
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"Wheel does not exist: {wheel}")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="vamos-clean-wheel-env-") as raw_root:
        root = Path(raw_root)
        environment_root = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment_root)
        python = environment_root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
            check=True,
            cwd=root,
            env=environment,
        )
        subprocess.run([str(python), str(Path(__file__).resolve())], check=True, cwd=root, env=environment)


def _cli(*arguments: str) -> dict[str, object]:
    executable = Path(sys.executable).with_name("vamos.exe" if sys.platform == "win32" else "vamos")
    completed = subprocess.run(
        [str(executable), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise AssertionError("CLI did not emit one JSON object.")
    return value


def _assert_exact(source: Path, replay: Path) -> None:
    stored = vamos.load_result(source)
    reproduced = vamos.load_result(replay)
    for role in ("F", "X"):
        left = getattr(stored, role)
        right = getattr(reproduced, role)
        if left is None or right is None:
            raise AssertionError(f"Missing mandatory array {role}.")
        if left.dtype.str != right.dtype.str or left.shape != right.shape:
            raise AssertionError(f"Array layout mismatch for {role}.")
        if np.ascontiguousarray(left).tobytes() != np.ascontiguousarray(right).tobytes():
            raise AssertionError(f"Array content mismatch for {role}.")


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


if __name__ == "__main__":
    main()

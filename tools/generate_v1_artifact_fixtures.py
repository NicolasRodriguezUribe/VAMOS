"""Capture permanent 1.0.0 artifact fixtures from a clean installed wheel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from release_smoke import _assert_no_personal_paths, _block_numba_import, _network_denied

import vamos


def _spec(*, engine: str, seeds: list[int]) -> vamos.StudySpec:
    return vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=seeds,
        max_evaluations=8,
        pop_size=4,
        engine=engine,
        on_error="fail_fast",
        max_attempts_per_task=2,
    )


def _command(root: Path, study: str, expected_exit: int) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "study", "inspect", study, "--json"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != expected_exit:
        raise AssertionError(f"Fixture CLI exited {result.returncode}: {result.stderr}")
    payload = json.loads(result.stdout)
    if payload["document_type"] != "vamos.study-command-result" or payload["schema_version"] != "1.0.0":
        raise AssertionError("Unexpected study command envelope")
    return payload


def _files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".npz"}:
            raise AssertionError(f"Unexpected fixture file: {path.name}")
        payload = path.read_bytes()
        encoding = "utf-8" if path.suffix == ".json" else "base64"
        content = payload.decode("utf-8") if encoding == "utf-8" else base64.b64encode(payload).decode("ascii")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "encoding": encoding,
                "content": content,
            }
        )
    return files


def capture() -> dict[str, Any]:
    if vamos.__version__ != "1.0.0" or "site-packages" not in Path(vamos.__file__).resolve().parts:
        raise AssertionError("Fixture capture requires a noneditable VAMOS 1.0.0 wheel installation")
    with tempfile.TemporaryDirectory(prefix="vamos-v1-artifact-fixtures-") as temporary:
        root = Path(temporary)
        with _network_denied():
            completed = vamos.create_study(_spec(engine="numpy", seeds=[0]), output=root / "study-completed").run()
            created = vamos.create_study(_spec(engine="numba", seeds=[10, 11]), output=root / "study-partial")
            with _block_numba_import():
                paused = created.run()
            partial = paused.resume()
        if completed.status != "completed" or partial.status != "completed_with_failures":
            raise AssertionError("Fixture studies did not reach the required states")
        failed = next(attempt for attempt in partial.attempts if attempt.status == "failed")
        succeeded = completed.attempts[0]
        if failed.run_reference is None or succeeded.run_reference is None:
            raise AssertionError("Fixture attempts are missing their canonical runs")
        failed_run = Path("study-partial") / str(failed.run_reference["path"])
        succeeded_run = Path("study-completed") / str(succeeded.run_reference["path"])
        for manifest in (failed_run, succeeded_run):
            vamos.load_run(root / manifest.parent, verify="all")
            document = json.loads((root / manifest).read_text(encoding="utf-8"))
            if document["provenance"]["source"].get("git_sha") is not None:
                raise AssertionError("Wheel fixtures must not capture a checkout revision")
        _assert_no_personal_paths(root)
        commands = {
            "completed": _command(root, "study-completed", 0),
            "completed_with_failures": _command(root, "study-partial", 6),
        }
        return {
            "document_type": "vamos.compatibility-artifact-fixtures",
            "schema_version": "1.0.0",
            "producer": {"repository": "vamos-optimization/VAMOS", "package": "vamos-optimization", "version": vamos.__version__},
            "fixtures": {
                "successful_run": succeeded_run.parent.as_posix(),
                "failed_run": failed_run.parent.as_posix(),
                "completed_study": "study-completed",
                "completed_with_failures_study": "study-partial",
            },
            "study_command_results": commands,
            "files": _files(root),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New fixture JSON destination; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite permanent fixtures: {args.output}")
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"VAMOS 1.0.0 artifact fixtures: {len(payload['files'])} files captured")


if __name__ == "__main__":
    main()

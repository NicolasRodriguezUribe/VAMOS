"""Deterministic public end-to-end smoke for an installed VAMOS wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import vamos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--mode", choices=("core", "full"), default="full")
    args = parser.parse_args()
    evidence = run_smoke(args.version, full=args.mode == "full")
    print(json.dumps(evidence, sort_keys=True))


def run_smoke(version: str, *, full: bool) -> dict[str, Any]:
    package_path = Path(vamos.__file__).resolve()
    if "site-packages" not in {part.lower() for part in package_path.parts}:
        raise AssertionError(f"Expected a noneditable wheel installation, got {package_path}")
    if vamos.__version__ != version:
        raise AssertionError(f"Expected VAMOS {version}, got {vamos.__version__}")
    removed_runtime_name = "Study" + "Runner"
    if hasattr(vamos, removed_runtime_name):
        raise AssertionError(f"The removed pre-public {removed_runtime_name} is exposed by the installed wheel.")
    with tempfile.TemporaryDirectory(prefix="vamos-release-smoke-") as raw_root:
        root = Path(raw_root)
        run_evidence = _run_lifecycle(root)
        study_evidence = _study_lifecycle(root)
        failure_evidence = _failure_lifecycle(root) if full else {"status": "not-run-in-core-mode"}
        cli_evidence = _cli_lifecycle(run_evidence["moved_root"], study_evidence["relocated_root"])
        _assert_no_personal_paths(root)
    return {
        "document_type": "vamos.release-smoke",
        "schema_version": "1.0.0",
        "status": "passed",
        "version": version,
        "mode": "full" if full else "core",
        "vamos_file": str(package_path),
        "network_policy": "denied-during-python-execution",
        "mutation_model": "single-owner-sequential",
        "run": {key: value for key, value in run_evidence.items() if not key.endswith("_root")},
        "study": {key: value for key, value in study_evidence.items() if not key.endswith("_root")},
        "failure_recovery": failure_evidence,
        "cli": cli_evidence,
    }


def _run_lifecycle(root: Path) -> dict[str, Any]:
    original = root / "run-original"
    moved = root / "run-moved"
    replayed = root / "run-replayed"
    with _network_denied():
        result = vamos.optimize(
            "zdt1",
            algorithm="nsgaii",
            pop_size=8,
            max_evaluations=16,
            engine="numpy",
            seed=7,
            n_var=6,
        )
        vamos.save_result(result, original)
        source_hashes = _snapshot(original)
        shutil.move(original, moved)
        run = vamos.load_run(moved)
        loaded = vamos.load_result(moved)
        verification = vamos.verify_run(moved, require_level="exact")
        replay = vamos.reproduce(moved, output=replayed)
        replayed_result = vamos.load_result(replayed)
    manifest_payload = json.loads((moved / "manifest.json").read_text(encoding="utf-8"))
    if (manifest_payload.get("document_type"), manifest_payload.get("schema_version")) != (
        "vamos.run-manifest",
        "1.0.0",
    ):
        raise AssertionError("The installed wheel did not write the public RunManifest 1.0 schema.")
    _same_array(loaded.F, replayed_result.F, "F")
    _same_array(loaded.X, replayed_result.X, "X")
    if not replay.exact or verification.effective_replayability != "exact":
        raise AssertionError("Run did not verify and replay exactly.")
    if _snapshot(moved) != source_hashes:
        raise AssertionError("Moving/loading/replaying changed the source run.")
    return {
        "run_id": run.manifest.run_id,
        "task_id": run.manifest.task_id,
        "evaluations": result.data["evaluations"],
        "exact_replay": True,
        "moved_root": moved,
    }


def _study_lifecycle(root: Path) -> dict[str, Any]:
    original = root / "study-original"
    relocated = root / "study-relocated"
    spec = vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=[0, 1],
        max_evaluations=16,
        pop_size=8,
        engine="numpy",
        on_error="continue",
    )
    with _network_denied():
        plan = vamos.plan_study(spec, output=original)
        if original.exists():
            raise AssertionError("Study planning wrote to the destination.")
        created = vamos.create_study(spec, output=original)
        if tuple(task.task_id for task in created.tasks) != plan.task_ids:
            raise AssertionError("Stable task identities changed between plan and creation.")
        completed = created.run()
        report = completed.inspect()
        summary = completed.summarize()
    manifest_payload = json.loads((original / "study-manifest.json").read_text(encoding="utf-8"))
    if (manifest_payload.get("document_type"), manifest_payload.get("schema_version")) != (
        "vamos.study-manifest",
        "1.0.0",
    ):
        raise AssertionError("The installed wheel did not write the public StudyManifest 1.0 schema.")
    if completed.status != "completed" or report.counts.get("succeeded") != len(plan.task_ids):
        raise AssertionError("Durable study did not complete every planned task.")
    _assert_summary_traceability(summary, plan.task_ids)
    successful_attempts = {task.task_id: tuple(task.attempts) for task in completed.tasks}
    resumed = completed.resume()
    if {task.task_id: tuple(task.attempts) for task in resumed.tasks} != successful_attempts:
        raise AssertionError("Resume repeated successful work.")
    _verify_referenced_runs(resumed)
    shutil.move(original, relocated)
    reloaded = vamos.load_study(relocated)
    relocated_report = reloaded.inspect()
    relocated_summary = reloaded.summarize()
    if relocated_report.counts != report.counts or [row.task_id for row in relocated_summary.rows] != [row.task_id for row in summary.rows]:
        raise AssertionError("Relocated study inspection or summary changed.")
    return {
        "study_id": completed.study_id,
        "plan_id": completed.plan_id,
        "task_ids": list(plan.task_ids),
        "tasks": len(plan.task_ids),
        "verified_runs": report.verified_run_count,
        "relocated": True,
        "relocated_root": relocated,
    }


def _failure_lifecycle(root: Path) -> dict[str, Any]:
    study_root = root / "study-failure-retry"
    spec = vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=[10, 11],
        max_evaluations=16,
        pop_size=8,
        engine="numba",
        on_error="fail_fast",
        max_attempts_per_task=2,
    )
    created = vamos.create_study(spec, output=study_root)
    with _block_numba_import(), _network_denied():
        paused = created.run()
    failed = [task for task in paused.tasks if task.state == "failed"]
    if paused.status != "paused" or len(failed) != 1:
        raise AssertionError("Controlled backend failure was not preserved as one paused task.")
    failed_task = failed[0]
    original_attempt = next(attempt for attempt in paused.attempts if attempt.task_id == failed_task.task_id)
    if original_attempt.failure is None or original_attempt.failure.get("retryable") is not True:
        raise AssertionError("Controlled task failure is not explicitly retryable.")
    if original_attempt.run_reference is None:
        raise AssertionError("Controlled task failure has no preserved canonical run.")
    original_path = study_root / str(original_attempt.run_reference["path"]).removesuffix("/manifest.json")
    original_hashes = _snapshot(original_path)
    with _network_denied():
        resumed = paused.resume()
    if resumed.status != "completed_with_failures":
        raise AssertionError("Resume did not finish eligible work while preserving the failed task.")
    succeeded_before_retry = {task.task_id: tuple(task.attempts) for task in resumed.tasks if task.state == "succeeded"}
    with _network_denied():
        completed = resumed.retry(failed_only=True)
    if completed.status != "completed":
        raise AssertionError("Explicit retry did not complete the study.")
    for task_id, attempts in succeeded_before_retry.items():
        current = next(task for task in completed.tasks if task.task_id == task_id)
        if tuple(current.attempts) != attempts:
            raise AssertionError("Explicit retry repeated a previously successful task.")
    lineage = [attempt for attempt in completed.attempts if attempt.task_id == failed_task.task_id]
    if [attempt.status for attempt in lineage] != ["failed", "succeeded"]:
        raise AssertionError("Failed-attempt lineage was not preserved across retry.")
    if _snapshot(original_path) != original_hashes:
        raise AssertionError("Retry modified the preserved failed canonical run.")
    _verify_referenced_runs(completed)
    _assert_summary_traceability(completed.summarize(), tuple(task.task_id for task in completed.tasks))
    return {
        "status": "passed",
        "failed_task_id": failed_task.task_id,
        "failed_attempt_id": original_attempt.attempt_id,
        "retry_attempt_id": lineage[1].attempt_id,
        "final_state": completed.status,
        "failed_attempt_preserved": True,
        "successful_tasks_repeated": False,
    }


def _cli_lifecycle(run_root: Path, study_root: Path) -> dict[str, Any]:
    human_commands = (
        ("--help",),
        ("results", "inspect", "--help"),
        ("results", "verify", "--help"),
        ("reproduce", "--help"),
        ("study", "plan", "--help"),
        ("study", "create", "--help"),
        ("study", "run", "--help"),
        ("study", "inspect", "--help"),
        ("study", "resume", "--help"),
        ("study", "retry", "--help"),
        ("study", "summarize", "--help"),
    )
    for arguments in human_commands:
        completed = _cli(*arguments)
        if "usage:" not in completed.stdout.lower():
            raise AssertionError(f"CLI help is missing usage text: {arguments}")
    json_commands = (
        ("results", "inspect", str(run_root), "--json"),
        ("results", "verify", str(run_root), "--require-level", "exact", "--json"),
        ("study", "inspect", str(study_root), "--json"),
        ("study", "summarize", str(study_root), "--format", "json", "--json"),
    )
    document_types: list[str] = []
    for arguments in json_commands:
        completed = _cli(*arguments)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise AssertionError(f"CLI JSON mode did not emit one object: {arguments}")
        document_types.append(str(payload.get("document_type")))
    failure_commands = (
        (("results", "inspect", "missing-release-smoke-run", "--json"), 4),
        (("study", "inspect", "missing-release-smoke-study", "--json"), 3),
    )
    failure_types: list[str] = []
    for arguments, expected_exit in failure_commands:
        completed = _cli(*arguments, expected_exit=expected_exit)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise AssertionError(f"CLI JSON failure did not emit one object: {arguments}")
        failure_types.append(str(payload.get("document_type")))
    usage = _cli("results", "inspect", expected_exit=2)
    if "usage:" not in usage.stderr.lower():
        raise AssertionError("CLI usage failure did not write usage guidance to stderr.")
    return {
        "human_commands": len(human_commands),
        "json_commands": len(json_commands),
        "document_types": document_types,
        "failure_exit_codes": [expected for _, expected in failure_commands],
        "failure_document_types": failure_types,
        "usage_exit_code": 2,
    }


def _cli(*arguments: str, expected_exit: int = 0) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("vamos.exe" if os.name == "nt" else "vamos")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != expected_exit:
        raise AssertionError(
            f"CLI command exited {completed.returncode}, expected {expected_exit}: {arguments}\n{completed.stdout}\n{completed.stderr}"
        )
    return completed


def _verify_referenced_runs(study: Any) -> None:
    for attempt in study.attempts:
        if attempt.run_reference is None:
            continue
        manifest = study.root / str(attempt.run_reference["path"])
        run = vamos.load_run(manifest.parent, verify="all")
        verification = vamos.verify_run(manifest.parent)
        if run.manifest.task_id != attempt.task_id or verification.artifact_integrity != "valid":
            raise AssertionError("A study run reference failed identity or integrity verification.")


def _assert_summary_traceability(summary: Any, task_ids: tuple[str, ...]) -> None:
    rows = {row.task_id: row for row in summary.rows}
    if set(rows) != set(task_ids):
        raise AssertionError("Study summary task identities differ from the plan.")
    for row in rows.values():
        if row.state == "succeeded" and not (row.selected_attempt_id and row.selected_run_id and row.run_manifest_sha256):
            raise AssertionError("Successful summary row lacks selected attempt/run traceability.")


def _same_array(left: np.ndarray | None, right: np.ndarray | None, role: str) -> None:
    if left is None or right is None:
        raise AssertionError(f"Missing {role} array.")
    if left.dtype.str != right.dtype.str or left.shape != right.shape:
        raise AssertionError(f"{role} layout changed during replay.")
    if np.ascontiguousarray(left).tobytes() != np.ascontiguousarray(right).tobytes():
        raise AssertionError(f"{role} bytes changed during replay.")


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_no_personal_paths(root: Path) -> None:
    patterns = (str(Path.home()), os.environ.get("USERPROFILE", ""))
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker and marker.lower() in text.lower() for marker in patterns):
            hits.append(path.relative_to(root).as_posix())
    if hits:
        raise AssertionError(f"Persisted artifacts contain personal paths: {sorted(hits)}")


@contextmanager
def _network_denied() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def denied(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("Network access is forbidden during release smoke execution.")

    socket.socket.connect = denied
    socket.create_connection = denied
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


class _NumbaImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del path, target
        if fullname == "numba" or fullname.startswith("numba."):
            raise ModuleNotFoundError("controlled release-smoke numba import failure", name=fullname)
        return None


@contextmanager
def _block_numba_import() -> Iterator[None]:
    loaded = sorted(name for name in sys.modules if name == "numba" or name.startswith("numba."))
    if loaded:
        raise AssertionError(f"Numba was imported before the controlled-failure phase: {loaded[:5]}")
    blocker = _NumbaImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


if __name__ == "__main__":
    main()

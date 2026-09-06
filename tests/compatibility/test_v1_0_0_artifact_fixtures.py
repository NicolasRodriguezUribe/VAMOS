from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pytest

import vamos

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "v1_0_0" / "artifact_fixtures.json"


def _tree(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def frozen_artifacts(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["document_type"] == "vamos.compatibility-artifact-fixtures"
    assert payload["schema_version"] == "1.0.0"
    assert payload["producer"] == {"repository": "vamos-optimization/VAMOS", "package": "vamos-optimization", "version": "1.0.0"}
    assert 1 <= len(payload["files"]) <= 200
    for entry in payload["files"]:
        relative = PurePosixPath(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative)
        target = tmp_path.joinpath(*relative.parts)
        assert target.resolve().is_relative_to(tmp_path.resolve()) and not target.exists()
        assert entry["encoding"] in {"utf-8", "base64"}
        content = entry["content"].encode("utf-8") if entry["encoding"] == "utf-8" else base64.b64decode(entry["content"], validate=True)
        assert len(content) == entry["bytes"] <= 1_000_000
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return tmp_path, payload


def test_permanent_successful_run_loads_exact_stored_arrays_without_mutation(frozen_artifacts: tuple[Path, dict[str, Any]]) -> None:
    root, fixture = frozen_artifacts
    run_root = root / fixture["fixtures"]["successful_run"]
    before = _tree(root)

    run = vamos.load_run(run_root, verify="all")
    result = vamos.load_result(run_root)
    verification = vamos.verify_run(run_root)

    assert run.status == "succeeded"
    assert verification.artifact_integrity == "valid"
    with np.load(run_root / "result.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(result.F, arrays["F"])
        np.testing.assert_array_equal(result.X, arrays["X"])
    assert _tree(root) == before


def test_permanent_failed_run_remains_inspectable_without_result_arrays(frozen_artifacts: tuple[Path, dict[str, Any]]) -> None:
    root, fixture = frozen_artifacts
    run_root = root / fixture["fixtures"]["failed_run"]
    before = _tree(root)

    run = vamos.load_run(run_root, verify="all")

    assert run.status == "failed"
    assert not (run_root / "result.npz").exists()
    assert _tree(root) == before


@pytest.mark.parametrize(
    ("name", "state", "succeeded", "failed"),
    [("completed_study", "completed", 1, 0), ("completed_with_failures_study", "completed_with_failures", 1, 1)],
)
def test_permanent_studies_load_inspect_and_summarize_without_execution(
    frozen_artifacts: tuple[Path, dict[str, Any]], name: str, state: str, succeeded: int, failed: int
) -> None:
    root, fixture = frozen_artifacts
    before = _tree(root)

    study = vamos.load_study(root / fixture["fixtures"][name])
    report = study.inspect()
    summary = study.summarize()

    assert study.status == report.state == state
    assert report.issues == ()
    assert report.counts["succeeded"] == succeeded
    assert report.counts["failed"] == failed
    assert len(summary.rows) == succeeded + failed
    assert sum(row.state == "succeeded" for row in summary.rows) == succeeded
    assert sum(row.state == "failed" for row in summary.rows) == failed
    assert _tree(root) == before


@pytest.mark.parametrize(
    ("name", "study", "exit_code"),
    [("completed", "completed_study", 0), ("completed_with_failures", "completed_with_failures_study", 6)],
)
def test_permanent_study_command_results_match_the_public_json_contract(
    frozen_artifacts: tuple[Path, dict[str, Any]], name: str, study: str, exit_code: int
) -> None:
    root, fixture = frozen_artifacts
    before = _tree(root)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "study", "inspect", fixture["fixtures"][study], "--json"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == fixture["study_command_results"][name]
    assert _tree(root) == before

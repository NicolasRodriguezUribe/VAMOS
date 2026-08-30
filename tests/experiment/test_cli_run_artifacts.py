from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts.jsonio import manifest_self_hash, sha256_file, stored_json_bytes
from vamos.experiment.cli.run_artifact_cli import run_reproduce, run_results


def _source(tmp_path: Path) -> Path:
    result = vamos.optimize("zdt1", algorithm="nsgaii", pop_size=6, max_evaluations=12, engine="numpy", seed=0, n_var=6)
    return vamos.save_result(result, tmp_path / "source").root


def _environment_mismatch(source: Path) -> None:
    environment_path = source / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["python"]["version"] = "3.11.0"
    environment_path.write_bytes(stored_json_bytes(environment))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = next(item for item in manifest["artifacts"] if item["role"] == "environment")
    descriptor["bytes"] = environment_path.stat().st_size
    descriptor["sha256"] = sha256_file(environment_path)
    manifest["integrity"]["manifest_sha256"] = manifest_self_hash(manifest)
    manifest_path.write_bytes(stored_json_bytes(manifest))


def test_results_inspect_human_and_json_are_manifest_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source(tmp_path)

    run_results(["inspect", str(source)])
    human = capsys.readouterr()
    run_results(["inspect", str(source), "--json"])
    machine = capsys.readouterr()
    run_results(["inspect", str(source), "--json"])
    repeated = capsys.readouterr()

    assert "Arrays (metadata only; values not loaded)" in human.out
    assert "Full artifact verification: not performed" in human.out
    assert human.err == ""
    payload = json.loads(machine.out)
    assert payload["document_type"] == "vamos.run-inspection"
    assert payload["optimization_executed"] is False
    assert payload["full_artifact_verification"] is False
    assert all(set(item) == {"dtype", "role", "shape"} for item in payload["arrays"])
    assert machine.out == repeated.out


def test_results_verify_human_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source(tmp_path)

    run_results(["verify", str(source), "--require-level", "exact"])
    human = capsys.readouterr()
    run_results(["verify", str(source), "--require-level", "exact", "--json"])
    machine = capsys.readouterr()

    assert "Effective replayability: exact" in human.out
    assert "Optimization executed: no" in human.out
    payload = json.loads(machine.out)
    assert payload["document_type"] == "vamos.verification-report"
    assert payload["effective_replayability"] == "exact"


def test_reproduce_json_uses_shared_python_service(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source(tmp_path)
    output = tmp_path / "replay"

    run_reproduce([str(source), "--output", str(output), "--json"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert payload["document_type"] == "vamos.replay-report"
    assert payload["exact"] is True
    assert Path(payload["output_root"]) == output
    assert vamos.load_run(output, verify="all").manifest["lineage"]["comparison"]["status"] == "exact_match"


def test_replay_lineage_inspection_survives_relocation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source(tmp_path)
    replay = vamos.reproduce(source, output=tmp_path / "replay")
    relocated = tmp_path / "relocated"
    (tmp_path / "replay").rename(relocated)

    run_results(["inspect", str(relocated), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["execution_kind"] == "replay"
    assert payload["lineage"]["source_run_id"] == replay.source_run_id
    assert payload["lineage"]["comparison"]["status"] == "exact_match"


def test_json_error_is_one_document_and_compatibility_exit_five(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _source(tmp_path)
    _environment_mismatch(source)

    with pytest.raises(SystemExit) as caught:
        run_results(["verify", str(source), "--require-level", "exact", "--json"])
    captured = capsys.readouterr()

    assert caught.value.code == 5
    payload = json.loads(captured.out)
    assert payload["document_type"] == "vamos.run-command-error"
    assert payload["error"]["optimization_executed"] is False
    assert captured.err == ""


def test_cli_exit_codes_for_integrity_schema_component_and_collision(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corrupt = _source(tmp_path / "corrupt")
    (corrupt / "result.npz").write_bytes(b"corrupt")
    with pytest.raises(SystemExit) as integrity:
        run_results(["verify", str(corrupt), "--json"])
    capsys.readouterr()

    future = _source(tmp_path / "future")
    manifest_path = future / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.0.0"
    manifest_path.write_bytes(stored_json_bytes(manifest))
    with pytest.raises(SystemExit) as schema:
        run_results(["verify", str(future), "--json"])
    capsys.readouterr()

    component = _source(tmp_path / "component")
    component_manifest = json.loads((component / "manifest.json").read_text(encoding="utf-8"))
    component_manifest["replayability"]["declared_level"] = "manual"
    component_manifest["replayability"]["exact_requirements"] = []
    component_manifest["integrity"]["manifest_sha256"] = manifest_self_hash(component_manifest)
    (component / "manifest.json").write_bytes(stored_json_bytes(component_manifest))
    with pytest.raises(SystemExit) as unavailable:
        run_reproduce([str(component), "--output", str(tmp_path / "unavailable"), "--json"])
    capsys.readouterr()

    collision_source = _source(tmp_path / "collision")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(SystemExit) as collision:
        run_reproduce([str(collision_source), "--output", str(occupied), "--json"])
    capsys.readouterr()

    assert integrity.value.code == 3
    assert schema.value.code == 4
    assert unavailable.value.code == 6
    assert collision.value.code == 8


def test_cli_usage_and_exact_mismatch_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as usage:
        run_results([])
    capsys.readouterr()

    source = _source(tmp_path)
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    original = replay_module._run_config

    def differing(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        changed = np.array(result.F, copy=True)
        changed.view(np.uint64).flat[0] ^= np.uint64(1)
        result.F = changed
        result.data["F"] = changed
        return result

    monkeypatch.setattr(replay_module, "_run_config", differing)
    with pytest.raises(SystemExit) as mismatch:
        run_reproduce([str(source), "--output", str(tmp_path / "mismatch"), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert usage.value.code == 2
    assert mismatch.value.code == 7
    assert payload["error"]["category"] == "replay_result_mismatch"
    assert payload["error"]["optimization_executed"] is True

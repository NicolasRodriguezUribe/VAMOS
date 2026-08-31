from __future__ import annotations

import importlib
import json
import os
import pickle
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

import vamos
from vamos.experiment.study.errors import (
    MalformedStudyError,
    StudyIntegrityError,
    StudyOutputCollisionError,
    StudyResourceLimitError,
    UnsafeStudyPathError,
)
from vamos.experiment.study.limits import StudyLoadLimits
from vamos.experiment.study.record_decoding import decode_attempt, decode_event
from vamos.experiment.study.serialization import seal_document, stored_document_bytes


def _spec(**changes: object) -> vamos.StudySpec:
    values: dict[str, object] = {
        "problems": ["zdt1"],
        "algorithms": ["nsgaii"],
        "seeds": [0],
        "max_evaluations": 24,
        "pop_size": 8,
    }
    values.update(changes)
    return vamos.StudySpec(**values)  # type: ignore[arg-type]


def _raw(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {item.relative_to(root).as_posix(): item.read_bytes() if item.is_file() else None for item in sorted(root.rglob("*"))}


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"junction creation failed: {completed.stdout} {completed.stderr}")
    else:
        os.symlink(target, link, target_is_directory=True)


def test_sa_015_and_sa_066_external_link_escape_is_rejected_without_skip(tmp_path: Path) -> None:
    root = tmp_path / "study"
    outside = tmp_path / "outside"
    vamos.create_study(_spec(), output=root)
    outside.mkdir()
    shutil.move(root / "events", outside / "events")
    link = root / "events"
    _make_directory_link(link, outside / "events")

    try:
        with pytest.raises(UnsafeStudyPathError) as caught:
            vamos.load_study(root)
        assert caught.value.reason == "UNSAFE_PATH"
    finally:
        if link.is_symlink():
            link.unlink()
        else:
            os.rmdir(link)


def test_existing_junction_or_symlink_destination_is_a_collision(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "study"
    _make_directory_link(link, target)
    try:
        with pytest.raises(StudyOutputCollisionError):
            vamos.create_study(_spec(), output=link)
        assert list(target.iterdir()) == []
    finally:
        os.rmdir(link)


def test_data_only_load_does_not_resolve_execute_import_pickle_shell_or_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(metadata={"plugin": "attacker.module:factory"}), output=root)
    before = _snapshot(root)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("executable operation invoked during data-only load")

    import vamos.experiment.study.planning as planning

    monkeypatch.setattr(planning, "_resolve_problem", forbidden)
    monkeypatch.setattr(vamos, "optimize", forbidden)
    monkeypatch.setattr(importlib, "import_module", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    loaded = vamos.load_study(root)
    assert loaded.status == "created"
    assert _snapshot(root) == before


def test_loading_applies_explicit_finite_resource_limits(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)

    with pytest.raises(StudyResourceLimitError) as caught:
        vamos.load_study(root, limits=StudyLoadLimits(max_plan_bytes=1))
    assert caught.value.reason == "RESOURCE_LIMIT"


def test_loaded_view_and_nested_values_are_immutable(tmp_path: Path) -> None:
    root = tmp_path / "study"
    loaded = vamos.create_study(_spec(metadata={"nested": {"items": [1, 2]}}), output=root)

    with pytest.raises(TypeError):
        loaded.spec.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.plan.tasks[0].resolved_run_spec["seed"] = 9  # type: ignore[index]
    assert loaded.spec.metadata["nested"]["items"] == (1, 2)
    assert vamos.load_study(root).plan_id == loaded.plan_id


def test_whitespace_reformat_is_detected_as_noncanonical_stored_bytes(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    task_path = next(root.glob("tasks/*/task.json"))
    task_path.write_text(json.dumps(_raw(task_path)), encoding="utf-8")

    with pytest.raises(StudyIntegrityError) as caught:
        vamos.load_study(root)
    assert caught.value.reason == "NON_CANONICAL_BYTES"


def test_invalid_task_id_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "study"
    vamos.create_study(_spec(), output=root)
    task_path = next(root.glob("tasks/*/task.json"))
    task = _raw(task_path)
    task["task_id"] = "sha256:" + "0" * 64
    task_path.write_bytes(stored_document_bytes(seal_document(task)))

    with pytest.raises(StudyIntegrityError) as caught:
        vamos.load_study(root)
    assert caught.value.reason == "TASK_ID_MISMATCH"


def test_attempt_schema_parses_without_creating_live_attempt() -> None:
    document = seal_document(
        {
            "document_type": "vamos.study-attempt",
            "schema_version": "1.0.0",
            "study_id": "11111111-1111-4111-8111-111111111111",
            "task_id": "sha256:" + "2" * 64,
            "attempt_id": "33333333-3333-4333-8333-333333333333",
            "attempt_number": 1,
            "execution_id": "22222222-2222-4222-8222-222222222222",
            "status": "created",
            "timestamps": {"created_at": "2026-08-31T12:00:00Z", "started_at": None, "completed_at": None},
            "lease_evidence": None,
            "failure": None,
            "run_reference": None,
            "integrity": {},
        }
    )
    attempt = decode_attempt(document)
    assert attempt.status == "created"
    assert attempt.attempt_number == 1


def test_initial_event_schema_and_integrity_are_valid(tmp_path: Path) -> None:
    root = tmp_path / "study"
    study = vamos.create_study(_spec(), output=root)
    event = decode_event(_raw(root / "events" / "00000000000000000001.json"))
    assert event.sequence == 1
    assert event.event_type == "study_created"
    assert event.entity_id == study.study_id


def test_attempt_unknown_field_is_rejected() -> None:
    document = {
        "document_type": "vamos.study-attempt",
        "schema_version": "1.0.0",
        "unexpected": True,
    }
    with pytest.raises(MalformedStudyError) as caught:
        decode_attempt(document)
    assert caught.value.reason == "UNKNOWN_FIELD"


def test_attempt_nested_timestamp_schema_is_closed() -> None:
    document = seal_document(
        {
            "document_type": "vamos.study-attempt",
            "schema_version": "1.0.0",
            "study_id": "11111111-1111-4111-8111-111111111111",
            "task_id": "sha256:" + "2" * 64,
            "attempt_id": "33333333-3333-4333-8333-333333333333",
            "attempt_number": 1,
            "execution_id": "22222222-2222-4222-8222-222222222222",
            "status": "created",
            "timestamps": {
                "created_at": "2026-08-31T12:00:00Z",
                "started_at": None,
                "completed_at": None,
                "unknown": True,
            },
            "lease_evidence": None,
            "failure": None,
            "run_reference": None,
            "integrity": {},
        }
    )
    with pytest.raises(MalformedStudyError) as caught:
        decode_attempt(document)
    assert caught.value.reason == "UNKNOWN_FIELD"

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from vamos import StudySpec, create_study, load_study
from vamos.experiment.study.errors import StudyError
from vamos.experiment.study.serialization import seal_document, stored_document_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "docs" / "dev" / "study_manifest_core_examples.json"


def _cases() -> list[dict[str, Any]]:
    payload = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    assert payload["document_type"] == "vamos.study-core-example-set"
    assert payload["schema_version"] == "1.0.0"
    return list(payload["cases"])


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["name"]))
def test_production_study_core_example(case: Mapping[str, Any], tmp_path: Path) -> None:
    operation = case["operation"]
    if operation == "create_load":
        _verify_valid(case, tmp_path)
    else:
        _verify_invalid(case, tmp_path)


def _verify_valid(case: Mapping[str, Any], tmp_path: Path) -> None:
    spec = StudySpec(**case["spec"])
    root = tmp_path / "source" / str(case["name"])
    created = create_study(spec, output=root)
    expected = case["expected"]
    if expected["relocate"]:
        moved = tmp_path / "relocated" / str(case["name"])
        moved.parent.mkdir()
        shutil.move(root, moved)
        root = moved
    loaded = load_study(root)
    assert loaded.status == expected["state"]
    assert len(loaded.tasks) == expected["tasks"]
    assert loaded.study_id == created.study_id
    assert loaded.plan_id == created.plan_id
    assert all(task.state == "pending" and not task.attempts for task in loaded.tasks)
    paths = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    assert not any(path.startswith("runs") or "attempts" in path for path in paths)


def _verify_invalid(case: Mapping[str, Any], tmp_path: Path) -> None:
    root = tmp_path / str(case["name"])
    create_study(
        StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], max_evaluations=24, pop_size=8),
        output=root,
    )
    mutation = case["mutation"]
    if mutation == "duplicate_json_key":
        (root / "study-manifest.json").write_bytes(b'{"document_type":"a","document_type":"b"}\n')
    elif mutation == "future_schema":
        path = root / "study-manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["schema_version"] = "2.0.0"
        path.write_bytes(stored_document_bytes(seal_document(value)))
    elif mutation == "task_id_mismatch":
        path = next(root.glob("tasks/*/task.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["task_id"] = "sha256:" + "0" * 64
        path.write_bytes(stored_document_bytes(seal_document(value)))
    else:
        raise AssertionError(f"unknown example mutation {mutation!r}")
    with pytest.raises(StudyError) as caught:
        load_study(root)
    assert caught.value.reason == case["expected_error"]

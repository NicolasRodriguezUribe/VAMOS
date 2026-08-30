from __future__ import annotations

import json
import re
from pathlib import Path

import vamos

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "docs" / "dev" / "run_artifact_contract.md"
ACCEPTANCE = REPO / "docs" / "dev" / "run_artifact_acceptance_tests.md"
EXAMPLES = REPO / "docs" / "dev" / "run_artifact_examples"


def test_contract_relative_links_resolve() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    links = re.findall(r"\]\((?!https?://)([^)#]+)", text)

    assert links
    assert all((CONTRACT.parent / link).resolve().exists() for link in links)
    assert "Pre-release run directories created before the canonical schema 1.0.0" in text


def test_acceptance_ids_are_contiguous_and_unique() -> None:
    text = ACCEPTANCE.read_text(encoding="utf-8")
    identifiers = [int(value) for value in re.findall(r"^\| RA-(\d{3}) \|", text, flags=re.MULTILINE)]

    assert identifiers == list(range(1, 32))
    referenced = {
        int(value)
        for path in Path(__file__).parent.glob("test_*.py")
        for value in re.findall(r"ra(\d{3})", path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
    }
    assert referenced <= set(identifiers)


def test_machine_readable_examples_match_canonical_layout() -> None:
    expected = {
        "custom-manual": "succeeded",
        "failed-run": "failed",
        "failed-replay": "failed",
        "moead-success": "succeeded",
        "nsgaii-success": "succeeded",
        "replay-mismatch": "succeeded",
        "replay-success": "succeeded",
    }
    assert {path.name for path in EXAMPLES.iterdir() if path.is_dir()} == set(expected)

    for name, status in expected.items():
        run = vamos.load_run(EXAMPLES / name, verify="all")
        assert run.status == status
        names = {path.name for path in run.root.iterdir()}
        if status == "succeeded":
            assert names == {"manifest.json", "result.npz", "environment.json"}
        else:
            assert names == {"manifest.json", "environment.json"}

    for report_name in ("verification-exact.json", "verification-incompatible.json"):
        report = json.loads((EXAMPLES / report_name).read_text(encoding="utf-8"))
        assert report["document_type"] == "vamos.verification-report"
        assert report["optimization_executed"] is False

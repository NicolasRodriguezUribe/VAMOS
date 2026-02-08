from __future__ import annotations

import json
from pathlib import Path

from vamos.assist.plan import create_plan
from vamos.assist.providers.mock_provider import MockPlanProvider


def test_auto_mode_writes_provider_trace_files(tmp_path: Path) -> None:
    plan_dir = create_plan(
        prompt="Create trace artifacts for provider request and response.",
        template=None,
        problem_type="real",
        out_dir=tmp_path / "assist_plan_auto_traces",
        mode="auto",
        provider=MockPlanProvider(),
        provider_name="mock",
    )

    request_path = plan_dir / "provider_request.json"
    response_path = plan_dir / "provider_response.json"
    assert request_path.is_file()
    assert response_path.is_file()

    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))

    assert isinstance(request, dict)
    assert isinstance(response, dict)

    for key in ("provider", "problem_type_hint", "templates_count", "catalog_summary", "allowed_override_keys_count"):
        assert key in request
    assert "prompt" not in request

    summary = request.get("catalog_summary")
    assert isinstance(summary, dict)
    for count_key in (
        "algorithms_count",
        "kernels_count",
        "crossover_methods_count",
        "mutation_methods_count",
        "templates_count",
    ):
        assert count_key in summary

    assert response.get("kind") in {"plan", "questions"}

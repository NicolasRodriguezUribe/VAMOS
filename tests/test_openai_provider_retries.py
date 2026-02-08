from __future__ import annotations

import json

import pytest

from vamos.assist.providers.openai_provider import OpenAIPlanProvider


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


def test_openai_provider_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIPlanProvider()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(provider, "_get_client", lambda: object())
    monkeypatch.setattr("vamos.assist.providers.openai_provider.time.sleep", lambda _: None)

    attempts: list[int] = []

    def _fake_responses_create(client: object, **kwargs: object) -> _FakeResponse:
        del client
        del kwargs
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("timeout while calling responses API")
        payload = {
            "kind": "plan",
            "template": "demo",
            "problem_type": "real",
            "overrides": {},
            "warnings": [],
        }
        return _FakeResponse(json.dumps(payload))

    monkeypatch.setattr(provider, "_responses_create", _fake_responses_create)

    result = provider.propose(
        prompt="Retry transient errors and return a plan.",
        catalog={"algorithms": ["nsgaii"], "kernels": ["numpy"]},
        templates=["demo"],
        problem_type_hint="real",
    )

    assert result["kind"] == "plan"
    assert result["template"] == "demo"
    assert len(attempts) == 3

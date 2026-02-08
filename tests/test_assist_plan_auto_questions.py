from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vamos.assist.plan import create_plan
from vamos.assist.providers.mock_provider import QuestionsMockProvider


def test_create_plan_auto_questions_noninteractive_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        create_plan(
            prompt="Need help selecting setup.",
            template=None,
            problem_type="real",
            out_dir=tmp_path / "assist_plan_auto_questions",
            mode="auto",
            provider=QuestionsMockProvider(),
            provider_name="mock",
        )

    message = str(excinfo.value)
    assert "Provider needs additional answers in non-interactive mode." in message
    assert "What problem family should this cover?" in message
    assert "Re-run interactively or use --mode template" in message

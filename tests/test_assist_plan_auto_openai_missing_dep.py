from __future__ import annotations

from pathlib import Path

import pytest

from vamos.assist.providers.openai_provider import OpenAIPlanProvider
from vamos.experiment.cli.quickstart import available_templates


def test_openai_provider_missing_sdk_raises_actionable_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    del tmp_path
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIPlanProvider()

    def _raise_import_error() -> object:
        raise ImportError("forced missing openai")

    monkeypatch.setattr(provider, "_import_openai", _raise_import_error)
    templates = available_templates()
    assert templates

    with pytest.raises(RuntimeError) as excinfo:
        provider.propose(
            prompt="Pick a template and budget.",
            catalog={"templates": templates},
            templates=templates,
            problem_type_hint="real",
        )
    message = str(excinfo.value)
    assert "pip install openai" in message
    assert "OPENAI_API_KEY" in message

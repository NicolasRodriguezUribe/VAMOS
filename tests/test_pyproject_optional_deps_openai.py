from __future__ import annotations

from pathlib import Path


def test_pyproject_has_openai_optional_dependency() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")

    try:
        import tomllib
    except ModuleNotFoundError:
        assert "[project.optional-dependencies]" in text
        assert "openai" in text
        return

    data = tomllib.loads(text)
    project = data.get("project", {})
    assert isinstance(project, dict)
    optional_deps = project.get("optional-dependencies", {})
    assert isinstance(optional_deps, dict)

    openai_deps = optional_deps.get("openai")
    assert isinstance(openai_deps, list)
    assert any(isinstance(item, str) and "openai" in item for item in openai_deps)

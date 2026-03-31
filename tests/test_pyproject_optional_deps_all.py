from __future__ import annotations

from pathlib import Path


def test_pyproject_all_extra_matches_union_of_other_optional_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        import tomli as tomllib  # type: ignore[import-not-found]

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    optional_deps = project["optional-dependencies"]

    all_extra = optional_deps["all"]
    expected_union: list[str] = []
    for group_name, deps in optional_deps.items():
        if group_name == "all":
            continue
        for dep in deps:
            if dep not in expected_union:
                expected_union.append(dep)

    assert all_extra == expected_union

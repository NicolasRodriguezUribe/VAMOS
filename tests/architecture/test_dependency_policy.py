from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None


BANNED_CORE = {
    "jax",
    "jmetalpy",
    "matplotlib",
    "moocore",
    "numba",
    "pymoo",
    "pygmo",
    "plotly",
    "seaborn",
    "scikit-learn",
    "sklearn",
    "streamlit",
    "tensorflow",
    "torch",
    "yellowbrick",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_name(dep: str) -> str:
    name = dep.strip()
    parts = re.split(r"[<>=!~\\[@ ]", name, maxsplit=1)
    return parts[0].strip().lower()


def test_dependency_policy() -> None:
    repo_root = _repo_root()
    pyproject = repo_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        project = data.get("project", {})
        deps = project.get("dependencies", [])
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            optional = {}
    else:
        deps = _extract_dependencies_from_text(text)
        optional = _extract_optional_dependencies_from_text(text)

    optional_lookup: dict[str, list[str]] = {}
    for group, group_deps in optional.items():
        for dep in group_deps:
            name = _normalize_name(dep)
            optional_lookup.setdefault(name, []).append(group)

    violations: list[str] = []
    for dep in deps:
        name = _normalize_name(dep)
        if name in BANNED_CORE:
            groups = ", ".join(sorted(optional_lookup.get(name, []))) or "none"
            violations.append(f"{name} found in [project].dependencies (optional groups: {groups})")

    if violations:
        details = "\n".join(f"- {item}" for item in sorted(violations))
        raise AssertionError(
            f"Core dependencies must not include optional/heavy libraries. Move them to [project.optional-dependencies].\n{details}"
        )


def _extract_dependencies_from_text(text: str) -> list[str]:
    deps: list[str] = []
    in_project = False
    in_deps = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            in_deps = False
            continue
        if not in_project:
            continue
        if line.startswith("dependencies"):
            if "[" in line:
                in_deps = True
                tail = line.split("[", 1)[1]
                if "]" in tail:
                    in_deps = False
                line = tail
            else:
                continue
        if in_deps:
            line = line.strip().rstrip(",")
            if line.startswith("]"):
                in_deps = False
                continue
            if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                deps.append(line.strip("\"'"))
            if "]" in line:
                in_deps = False
    return deps


def _extract_optional_dependencies_from_text(text: str) -> dict[str, list[str]]:
    optional: dict[str, list[str]] = {}
    in_optional = False
    current_group: str | None = None
    in_list = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_optional = line == "[project.optional-dependencies]"
            current_group = None
            in_list = False
            continue
        if not in_optional:
            continue
        if "=" in line and not in_list:
            group, rest = [part.strip() for part in line.split("=", 1)]
            current_group = group
            if "[" in rest:
                in_list = True
                payload = rest.split("[", 1)[1]
                optional.setdefault(current_group, []).extend(_extract_quoted_strings(payload))
                if "]" in payload:
                    in_list = False
                    current_group = None
            continue
        if in_list and current_group:
            optional.setdefault(current_group, []).extend(_extract_quoted_strings(line))
            if "]" in line:
                in_list = False
                current_group = None
    return optional


def _extract_quoted_strings(payload: str) -> list[str]:
    matches = re.findall(r"[\"']([^\"']+)[\"']", payload)
    return list(matches)

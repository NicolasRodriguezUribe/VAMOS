from __future__ import annotations

import re
from pathlib import Path


REQUIRED_LINK = "docs/dev/architecture_health.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_agents_health_link() -> None:
    repo_root = _repo_root()
    agent_files = sorted(repo_root.rglob("AGENTS.md"))
    rel_paths = [path.relative_to(repo_root).as_posix() for path in agent_files]

    failures: list[str] = []
    for path, rel_path in zip(agent_files, rel_paths, strict=False):
        text = path.read_text(encoding="utf-8-sig")
        missing = []
        if REQUIRED_LINK not in text:
            missing.append("missing link")
        if "Architecture Health" not in text and re.search(r"architecture health", text, re.IGNORECASE) is None:
            missing.append("missing header")
        if missing:
            failures.append(f"{rel_path}: {', '.join(missing)}")

    if failures:
        lines = ["AGENTS.md architecture health link check failed."]
        lines.append("Discovered AGENTS.md files:")
        lines.extend(f"- {path}" for path in rel_paths)
        lines.append("Failures:")
        lines.extend(f"- {item}" for item in failures)
        raise AssertionError("\n".join(lines))

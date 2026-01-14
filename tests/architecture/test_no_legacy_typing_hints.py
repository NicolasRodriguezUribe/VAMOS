from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = (
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "examples",
    REPO_ROOT / "experiments" / "scripts",
)
PATTERNS = (
    re.compile(r"\b(Optional|Union|List|Dict|Tuple|Set|FrozenSet)\["),
    re.compile(r"typing\.(Optional|Union|List|Dict|Tuple|Set|FrozenSet)"),
)


def _has_legacy_typing(text: str) -> bool:
    return any(pattern.search(text) for pattern in PATTERNS)


def test_no_legacy_typing_hints() -> None:
    offenders: list[str] = []
    for root in SCAN_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if _has_legacy_typing(text):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert not offenders, "Legacy typing hints found:\n" + "\n".join(offenders)

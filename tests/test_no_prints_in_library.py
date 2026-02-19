from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_PREFIXES = (
    "src/vamos/experiment/cli/",
    "src/vamos/experiment/scripts/",
    "src/vamos/ux/studio/",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _allowed_path(rel_path: str) -> bool:
    return rel_path.startswith(ALLOWED_PREFIXES)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def test_no_prints_in_library() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        if _allowed_path(rel_path):
            continue

        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in {"print", "pprint", "pprint.pprint"}:
                lineno = getattr(node, "lineno", "?")
                violations.append(f"{rel_path}:{lineno}: {name}()")

    if violations:
        msg = ["print() is forbidden outside CLI/UI modules:"]
        msg.extend(f"- {item}" for item in sorted(violations))
        raise AssertionError("\n".join(msg))

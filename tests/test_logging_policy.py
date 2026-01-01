from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_BASICCONFIG_PATHS: tuple[str, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_basic_config_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "logging":
            return func.attr == "basicConfig"
        return func.attr == "basicConfig"
    if isinstance(func, ast.Name):
        return func.id == "basicConfig"
    return False


def test_logging_policy() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        if rel_path in ALLOWED_BASICCONFIG_PATHS:
            continue

        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_basic_config_call(node):
                lineno = getattr(node, "lineno", "?")
                violations.append(f"{rel_path}:{lineno}: logging.basicConfig")

    if violations:
        msg = ["logging.basicConfig is forbidden in library modules:"]
        msg.extend(f"- {item}" for item in sorted(violations))
        raise AssertionError("\n".join(msg))

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_PREFIXES = (
    "vamos.api",
    "vamos.foundation.",
    "vamos.engine.api",
    "vamos.experiment.quick",
    "vamos.ux.api",
)
MAX_ALL_SIZE = 25


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_relative(module: str | None, level: int, name: str | None = None) -> str | None:
    if level <= 0:
        return module
    if level != 1:
        return None
    if module:
        return f"vamos.{module}"
    if name:
        return f"vamos.{name}"
    return "vamos"


def _allowed(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _extract_all_len(tree: ast.Module) -> int | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _literal_len(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "__all__":
                return _literal_len(node.value)
    return None


def _literal_len(node: ast.AST | None) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    return None


def test_public_api_guard() -> None:
    init_path = _repo_root() / "src" / "vamos" / "__init__.py"
    text = init_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)

    wildcard_imports: list[str] = []
    disallowed_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("vamos."):
                    if not _allowed(name):
                        disallowed_imports.append(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    wildcard_imports.append(node.module or "<relative>")

            module = _resolve_relative(node.module, node.level)
            if module and module.startswith("vamos.") and not _allowed(module):
                disallowed_imports.append(module)
            if node.module is None and node.level == 1:
                for alias in node.names:
                    resolved = _resolve_relative(None, 1, alias.name)
                    if resolved and resolved.startswith("vamos.") and not _allowed(resolved):
                        disallowed_imports.append(resolved)

    all_len = _extract_all_len(tree)
    errors: list[str] = []

    if wildcard_imports:
        errors.append(f"Wildcard imports are forbidden: {sorted(set(wildcard_imports))}")
    if disallowed_imports:
        errors.append(f"Disallowed imports: {sorted(set(disallowed_imports))}")
    if all_len is None:
        errors.append("__all__ must be defined as a list/tuple literal.")
    elif all_len > MAX_ALL_SIZE:
        errors.append(f"__all__ too large: {all_len} > {MAX_ALL_SIZE}")

    if errors:
        raise AssertionError("\n".join(errors))

from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_type_checking_if(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _is_main_if(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comp = test.comparators[0]
    return isinstance(comp, ast.Constant) and comp.value == "__main__"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return type(node).__name__


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((node.lineno, _call_name(node.func)))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _top_level_calls(tree: ast.Module) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []

    def walk(nodes: list[ast.stmt]) -> None:
        for stmt in nodes:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, ast.If):
                if _is_type_checking_if(stmt) or _is_main_if(stmt):
                    continue
                walk(stmt.body)
                walk(stmt.orelse)
                continue
            visitor = _CallCollector()
            visitor.visit(stmt)
            calls.extend(visitor.calls)

    walk(list(tree.body))
    return sorted(calls)


def test_no_import_time_side_effects() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []
    allowed_calls = {"TypeVar", "typing.TypeVar"}

    for path in sorted(src_root.rglob("*.py")):
        rel_path = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc
        for lineno, name in _top_level_calls(tree):
            if name in allowed_calls:
                continue
            violations.append(f"{rel_path}:{lineno}: call {name}")

    if violations:
        lines = ["Import-time side effects detected:"]
        lines.extend(f"- {item}" for item in sorted(violations))
        raise AssertionError("\n".join(lines))

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "vamos"
EXPERIMENT_ROOT = SRC_ROOT / "experiment"


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return "vamos." + ".".join(parts)


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


def _top_level_imports(tree: ast.Module) -> list[ast.stmt]:
    imports: list[ast.stmt] = []

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
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)

    walk(list(tree.body))
    return imports


def _resolve_relative(current: str, module: str | None, level: int) -> str | None:
    if level == 0:
        return module
    parts = current.split(".")
    if len(parts) < 2:
        return None
    package = parts[:-1]
    if level > len(package):
        return None
    base_parts = package[: len(package) - level + 1]
    if module:
        base_parts += module.split(".")
    return ".".join(base_parts)


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        visiting.add(node)
        path.append(node)
        for dep in sorted(graph.get(node, set())):
            if dep in visiting:
                idx = path.index(dep)
                return path[idx:] + [dep]
            if dep not in visited:
                cycle = dfs(dep)
                if cycle:
                    return cycle
        visiting.remove(node)
        path.pop()
        visited.add(node)
        return None

    for node in sorted(graph):
        if node in visited:
            continue
        cycle = dfs(node)
        if cycle:
            return cycle
    return None


def test_no_experiment_import_cycles() -> None:
    files = sorted(EXPERIMENT_ROOT.rglob("*.py"))
    modules = {_module_name(path) for path in files}
    graph: dict[str, set[str]] = {module: set() for module in modules}

    for path in files:
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for stmt in _top_level_imports(tree):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    name = alias.name
                    if name in modules:
                        graph[module].add(name)
            elif isinstance(stmt, ast.ImportFrom):
                target = _resolve_relative(module, stmt.module, stmt.level)
                if target is None:
                    continue
                if target in modules:
                    graph[module].add(target)
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    expanded = f"{target}.{alias.name}"
                    if expanded in modules:
                        graph[module].add(expanded)

    cycle = _find_cycle(graph)
    if cycle:
        pretty = " -> ".join(cycle)
        raise AssertionError(f"Experiment import cycle detected: {pretty}")

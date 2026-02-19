from __future__ import annotations

import ast
from pathlib import Path

BLACKLIST = {
    "pymoo",
    "jmetalpy",
    "pygmo",
    "streamlit",
    "torch",
    "tensorflow",
    "jax",
    "sklearn",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_importerror_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
        return True
    if isinstance(handler.type, ast.Tuple):
        return any(isinstance(elt, ast.Name) and elt.id == "ImportError" for elt in handler.type.elts)
    return False


def _iter_top_level_imports(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, False
        elif isinstance(node, ast.Try):
            guarded = any(_is_importerror_handler(h) for h in node.handlers)
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    yield sub, guarded


def _iter_top_level_statements(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.Try):
            guarded = any(_is_importerror_handler(h) for h in node.handlers)
            for sub in node.body:
                yield sub, guarded
        else:
            yield node, False


def _root_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            return alias.name.split(".", 1)[0]
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            return None
        if node.module:
            return node.module.split(".", 1)[0]
    return None


def _importlib_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    importlib_names: set[str] = set()
    import_module_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or "import_module")
    return importlib_names, import_module_names


def _dynamic_import_root(call: ast.Call, importlib_names: set[str], import_module_names: set[str]) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr != "import_module":
            return None
        if not isinstance(func.value, ast.Name):
            return None
        if func.value.id not in importlib_names:
            return None
    elif isinstance(func, ast.Name):
        if func.id == "__import__":
            pass
        elif func.id in import_module_names:
            pass
        else:
            return None
    else:
        return None
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.split(".", 1)[0]
    if isinstance(arg, ast.Str):
        return arg.s.split(".", 1)[0]
    return None


def test_optional_deps_policy() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    violations: list[str] = []

    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc

        in_foundation = rel_path.startswith("src/vamos/foundation/")
        in_engine = rel_path.startswith("src/vamos/engine/")
        in_external = rel_path.startswith("src/vamos/experiment/external/")
        in_studio = rel_path.startswith("src/vamos/ux/studio/")

        importlib_names, import_module_names = _importlib_aliases(tree)

        for node, guarded in _iter_top_level_imports(tree):
            root = _root_module(node)
            if root not in BLACKLIST:
                continue
            lineno = getattr(node, "lineno", "?")

            if in_studio:
                if root == "streamlit":
                    continue
                violations.append(f"{rel_path}:{lineno}: {root} (studio disallows)")
                continue

            if in_external:
                if guarded:
                    continue
                violations.append(f"{rel_path}:{lineno}: {root} (external requires try/except ImportError)")
                continue

            if in_foundation or in_engine:
                violations.append(f"{rel_path}:{lineno}: {root} (foundation/engine)")
                continue

            violations.append(f"{rel_path}:{lineno}: {root} (top-level import not allowed)")

        for stmt, guarded in _iter_top_level_statements(tree):
            for call in ast.walk(stmt):
                if not isinstance(call, ast.Call):
                    continue
                root = _dynamic_import_root(call, importlib_names, import_module_names)
                if root not in BLACKLIST:
                    continue
                lineno = getattr(call, "lineno", "?")
                if in_studio:
                    if root == "streamlit":
                        continue
                    violations.append(f"{rel_path}:{lineno}: {root} (studio dynamic import)")
                    continue
                if in_external:
                    if guarded:
                        continue
                    violations.append(f"{rel_path}:{lineno}: {root} (external requires try/except ImportError)")
                    continue
                if in_foundation or in_engine:
                    violations.append(f"{rel_path}:{lineno}: {root} (foundation/engine)")
                    continue
                violations.append(f"{rel_path}:{lineno}: {root} (top-level import not allowed)")

    if violations:
        msg = ["Optional dependency policy violations:"]
        msg.extend(f"- {item}" for item in sorted(set(violations)))
        raise AssertionError("\n".join(msg))

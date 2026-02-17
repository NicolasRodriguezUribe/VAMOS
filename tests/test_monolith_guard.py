from __future__ import annotations

import ast
import json
from pathlib import Path


CORE_LOC_MAX = 450
CLI_UI_LOC_MAX = 350
FUNC_LOC_MAX = 250
CLASS_LOC_MAX = 400

BUDGET_PATH = Path(__file__).resolve().parent / "architecture" / "monolith_guard_budget.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _count_nonblank(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _category_for(rel_path: str) -> str:
    if rel_path.startswith("src/vamos/foundation/") or rel_path.startswith("src/vamos/engine/"):
        return "core"
    if rel_path.startswith("src/vamos/experiment/cli/") or rel_path.startswith("src/vamos/ux/studio/"):
        return "cli_ui"
    return "core"


def _loc_threshold(category: str) -> int:
    return CLI_UI_LOC_MAX if category == "cli_ui" else CORE_LOC_MAX


def _node_span(node: ast.AST) -> tuple[int, int] | None:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        return None
    return lineno, end_lineno


def _top_defs(defs: list[tuple[int, str, int]], limit: int = 3) -> list[tuple[int, str, int]]:
    return sorted(defs, key=lambda item: (-item[0], item[1], item[2]))[:limit]


def _load_budget() -> dict[str, dict[str, int]]:
    if not BUDGET_PATH.exists():
        raise AssertionError(f"Missing monolith budget file: {BUDGET_PATH}")
    payload = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    return {
        "file_violations": dict(payload.get("file_violations", {})),
        "function_violations": dict(payload.get("function_violations", {})),
        "class_violations": dict(payload.get("class_violations", {})),
    }


def test_monolith_guard() -> None:
    repo_root = _repo_root()
    src_root = repo_root / "src" / "vamos"
    file_violations: list[tuple[str, int, int, str]] = []
    func_violations: list[tuple[str, str, int]] = []
    class_violations: list[tuple[str, str, int]] = []
    file_funcs: dict[str, list[tuple[int, str, int]]] = {}
    file_classes: dict[str, list[tuple[int, str, int]]] = {}

    for path in _iter_python_files(src_root):
        rel_path = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        loc = _count_nonblank(text)
        category = _category_for(rel_path)
        threshold = _loc_threshold(category)
        if loc > threshold:
            file_violations.append((rel_path, loc, threshold, category))

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover - should not happen
            raise AssertionError(f"Failed to parse {rel_path}: {exc}") from exc

        file_funcs[rel_path] = []
        file_classes[rel_path] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = _node_span(node)
                if span is None:
                    continue
                size = span[1] - span[0] + 1
                file_funcs[rel_path].append((size, node.name, span[0]))
                if size > FUNC_LOC_MAX:
                    func_violations.append((rel_path, node.name, size))
            elif isinstance(node, ast.ClassDef):
                span = _node_span(node)
                if span is None:
                    continue
                size = span[1] - span[0] + 1
                file_classes[rel_path].append((size, node.name, span[0]))
                if size > CLASS_LOC_MAX:
                    class_violations.append((rel_path, node.name, size))

    if not (file_violations or func_violations or class_violations):
        return

    budget = _load_budget()
    budget_files = {str(k): int(v) for k, v in budget["file_violations"].items()}
    budget_funcs = {str(k): int(v) for k, v in budget["function_violations"].items()}
    budget_classes = {str(k): int(v) for k, v in budget["class_violations"].items()}

    current_files = {path: int(loc) for path, loc, _threshold, _category in file_violations}
    current_funcs = {f"{path}:{name}": int(size) for path, name, size in func_violations}
    current_classes = {f"{path}:{name}": int(size) for path, name, size in class_violations}

    new_files = sorted(path for path in current_files if path not in budget_files)
    regressed_files = sorted(path for path, loc in current_files.items() if loc > int(budget_files.get(path, -1)))

    new_funcs = sorted(key for key in current_funcs if key not in budget_funcs)
    regressed_funcs = sorted(key for key, size in current_funcs.items() if size > int(budget_funcs.get(key, -1)))

    new_classes = sorted(key for key in current_classes if key not in budget_classes)
    regressed_classes = sorted(key for key, size in current_classes.items() if size > int(budget_classes.get(key, -1)))

    if not (new_files or regressed_files or new_funcs or regressed_funcs or new_classes or regressed_classes):
        return

    lines = ["Monolith guard violations detected:"]
    if new_files:
        lines.append("New file-size violations:")
        for key in new_files:
            lines.append(f"- {key}: {current_files[key]} LOC")
    if regressed_files:
        lines.append("Regressed file-size violations:")
        for key in regressed_files:
            lines.append(f"- {key}: {current_files[key]} LOC > budget {budget_files.get(key, '?')}")
    if new_funcs:
        lines.append("New function-size violations:")
        for key in new_funcs:
            lines.append(f"- {key}: {current_funcs[key]} LOC")
    if regressed_funcs:
        lines.append("Regressed function-size violations:")
        for key in regressed_funcs:
            lines.append(f"- {key}: {current_funcs[key]} LOC > budget {budget_funcs.get(key, '?')}")
    if new_classes:
        lines.append("New class-size violations:")
        for key in new_classes:
            lines.append(f"- {key}: {current_classes[key]} LOC")
    if regressed_classes:
        lines.append("Regressed class-size violations:")
        for key in regressed_classes:
            lines.append(f"- {key}: {current_classes[key]} LOC > budget {budget_classes.get(key, '?')}")

    lines.append("")
    lines.append("Detailed current violations:")
    if file_violations:
        lines.append("File size violations:")
        for rel_path, loc, threshold, category in sorted(file_violations):
            lines.append(f"- {rel_path}: {loc} LOC > {threshold} ({category})")
            top_funcs = _top_defs(file_funcs.get(rel_path, []))
            top_classes = _top_defs(file_classes.get(rel_path, []))
            if top_funcs:
                lines.append("  top functions:")
                for size, name, lineno in top_funcs:
                    lines.append(f"    - {name} (line {lineno}, {size} LOC)")
            else:
                lines.append("  top functions: none")
            if top_classes:
                lines.append("  top classes:")
                for size, name, lineno in top_classes:
                    lines.append(f"    - {name} (line {lineno}, {size} LOC)")
            else:
                lines.append("  top classes: none")
    if func_violations:
        lines.append("Function size violations:")
        for rel_path, name, size in sorted(func_violations):
            lines.append(f"- {rel_path}:{name}: {size} LOC > {FUNC_LOC_MAX}")
    if class_violations:
        lines.append("Class size violations:")
        for rel_path, name, size in sorted(class_violations):
            lines.append(f"- {rel_path}:{name}: {size} LOC > {CLASS_LOC_MAX}")
    lines.append("Hint: Split into a package with focused modules; keep orchestration thin.")

    raise AssertionError("\n".join(lines))

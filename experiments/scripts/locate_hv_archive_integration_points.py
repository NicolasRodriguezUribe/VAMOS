from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

REPO = Path.cwd()
SRC = REPO / "src" / "vamos"
OUT = REPO / "experiments" / "catalog" / "integration_targets.json"

LOOP_HINTS = ("max_evaluations", "evaluations", "n_evals", "offspring", "population", "generation", "iterate", "step")
WRITE_HINTS = ("metadata.json", "FUN.csv", "time.txt", "resolved_config.json", "write_text", "to_csv", "open(", "Path(")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def extract_snippet(lines: list[str], lineno: int, ctx: int = 6) -> str:
    lo = max(1, lineno - ctx)
    hi = min(len(lines), lineno + ctx)
    chunk = []
    for i in range(lo, hi + 1):
        chunk.append(f"{i:5d}: {lines[i - 1].rstrip()}")
    return "\n".join(chunk)


class LoopVisitor(ast.NodeVisitor):
    def __init__(self):
        self.candidates: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._scan_body(node, kind="function", name=node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._scan_body(node, kind="async_function", name=node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        for b in node.body:
            if isinstance(b, ast.FunctionDef):
                self._scan_body(b, kind="method", name=f"{node.name}.{b.name}")
        self.generic_visit(node)

    def _scan_body(self, node: ast.AST, kind: str, name: str) -> None:
        # Heuristic scoring: loops + mentions of max_evaluations/evaluations
        score = 0
        loop_nodes = 0
        name_hits = 0

        for sub in ast.walk(node):
            if isinstance(sub, (ast.For, ast.While)):
                loop_nodes += 1
            if isinstance(sub, ast.Name) and sub.id in LOOP_HINTS:
                name_hits += 1
            if isinstance(sub, ast.Attribute) and sub.attr in LOOP_HINTS:
                name_hits += 1

            # while ... < max_evaluations
            if isinstance(sub, ast.Compare):
                txt = ast.unparse(sub) if hasattr(ast, "unparse") else ""
                if "max_evaluations" in txt and ("<" in txt or "<=" in txt):
                    score += 8
                if "evaluations" in txt and ("<" in txt or "<=" in txt):
                    score += 5

        score += 2 * loop_nodes + 1 * name_hits
        if score >= 8:  # threshold
            self.candidates.append(
                {
                    "kind": kind,
                    "name": name,
                    "score": score,
                    "lineno": getattr(node, "lineno", None),
                }
            )


def scan_file(path: Path) -> dict[str, Any]:
    txt = read_text(path)
    lines = txt.splitlines()
    res: dict[str, Any] = {"path": str(path.relative_to(REPO)), "loop_candidates": [], "write_hits": []}

    # write-hits (string scan)
    for i, line in enumerate(lines, start=1):
        if any(h in line for h in WRITE_HINTS):
            if ("metadata.json" in line) or ("FUN.csv" in line) or ("time.txt" in line) or ("resolved_config.json" in line):
                res["write_hits"].append({"lineno": i, "line": line.strip(), "snippet": extract_snippet(lines, i)})

    # AST loop candidates
    try:
        tree = ast.parse(txt)
        v = LoopVisitor()
        v.visit(tree)
        for c in v.candidates:
            ln = c.get("lineno") or 1
            c["snippet"] = extract_snippet(lines, int(ln))
        res["loop_candidates"] = v.candidates
    except Exception as e:
        res["parse_error"] = str(e)

    return res


def main() -> int:
    files = sorted(SRC.rglob("*.py"))
    targets: list[dict[str, Any]] = []

    for p in files:
        r = scan_file(p)
        if r["loop_candidates"] or r["write_hits"]:
            targets.append(r)

    # Rank: files with loop_candidates first, then write_hits
    def keyfn(r: dict[str, Any]) -> tuple[int, int]:
        return (len(r.get("loop_candidates", [])), len(r.get("write_hits", [])))

    targets.sort(key=keyfn, reverse=True)

    OUT.write_text(json.dumps({"targets": targets}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", OUT.relative_to(REPO))
    print("Files with hits:", len(targets))

    # Console summary: top 12 loop files + top 8 write-hit files
    loop_files = [t for t in targets if t.get("loop_candidates")]
    write_files = [t for t in targets if t.get("write_hits")]

    print("\nTop loop-candidate files:")
    for t in loop_files[:12]:
        best = sorted(t["loop_candidates"], key=lambda x: -x["score"])[0]
        print(f"  {t['path']}  (n={len(t['loop_candidates'])}, best_score={best['score']}, best={best['name']})")

    print("\nTop artifact-writer hint files:")
    for t in write_files[:8]:
        print(f"  {t['path']}  (write_hits={len(t['write_hits'])})")

    # Print first concrete snippets for the best file of each category
    if loop_files:
        t = loop_files[0]
        best = sorted(t["loop_candidates"], key=lambda x: -x["score"])[0]
        print("\n=== BEST LOOP SNIPPET ===")
        print(t["path"])
        print(best["name"], "score=", best["score"])
        print(best["snippet"])

    if write_files:
        t = write_files[0]
        print("\n=== BEST WRITER SNIPPET ===")
        print(t["path"])
        print(t["write_hits"][0]["snippet"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

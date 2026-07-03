"""
Generate reproducible accessibility-proxy tables for the VAMOS manuscript.

The script reads canonical onboarding snippets from
``paper/accessibility_proxy_snippets.json`` and writes two LaTeX artifacts:

* ``paper/manuscript/accessibility_proxies.tex`` for the main paper
* ``paper/manuscript/accessibility_proxy_details.tex`` for the supplementary

The compact table reports values as ``LOC (imports)``, where LOC counts
non-empty, non-comment lines in each canonical snippet.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT_DIR / "paper"
MANUSCRIPT_DIR = PAPER_DIR / "manuscript"
DEFAULT_DATASET = PAPER_DIR / "accessibility_proxy_snippets.json"
DEFAULT_MAIN_TABLE = MANUSCRIPT_DIR / "accessibility_proxies.tex"
DEFAULT_DETAILS_TABLE = MANUSCRIPT_DIR / "accessibility_proxy_details.tex"


@dataclass(frozen=True)
class SnippetMetrics:
    framework: str
    task_id: str
    task_label: str
    loc: int
    imports: int
    guided_tools: bool
    entry_shape: str
    source_label: str
    source_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate accessibility-proxy LaTeX tables.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the canonical snippet dataset.")
    parser.add_argument("--main-out", type=Path, default=DEFAULT_MAIN_TABLE, help="Output path for the main manuscript table.")
    parser.add_argument(
        "--details-out",
        type=Path,
        default=DEFAULT_DETAILS_TABLE,
        help="Output path for the supplementary detailed table.",
    )
    parser.add_argument("--compile-main", action="store_true", help="Compile paper/manuscript/main.tex after writing tables.")
    parser.add_argument(
        "--compile-supplementary",
        action="store_true",
        help="Compile paper/manuscript/supplementary.tex after writing tables.",
    )
    return parser.parse_args()


def _nonempty_code_lines(code: str) -> list[str]:
    lines: list[str] = []
    for raw in code.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(raw.rstrip())
    return lines


def count_loc(code: str) -> int:
    return len(_nonempty_code_lines(code))


def count_imports(code: str) -> int:
    return sum(1 for line in _nonempty_code_lines(code) if re.match(r"^\s*(import|from)\s+\S+", line))


def load_metrics(dataset_path: Path) -> tuple[dict[str, str], list[str], list[SnippetMetrics]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    task_order = payload["task_order"]
    task_labels = {task_id: payload["tasks"][task_id]["label"] for task_id in task_order}

    metrics: list[SnippetMetrics] = []
    for framework in payload["frameworks"]:
        name = framework["name"]
        guided_tools = bool(framework["guided_tools"])
        snippets = framework["snippets"]
        for task_id in task_order:
            snippet = snippets[task_id]
            code = snippet["code"]
            metrics.append(
                SnippetMetrics(
                    framework=name,
                    task_id=task_id,
                    task_label=task_labels[task_id],
                    loc=count_loc(code),
                    imports=count_imports(code),
                    guided_tools=guided_tools,
                    entry_shape=snippet["entry_shape"],
                    source_label=snippet["source_label"],
                    source_url=snippet["source_url"],
                )
            )
    return task_labels, task_order, metrics


def _framework_order(metrics: list[SnippetMetrics]) -> list[str]:
    seen: list[str] = []
    for item in metrics:
        if item.framework not in seen:
            seen.append(item.framework)
    return seen


def _task_min_locs(metrics: list[SnippetMetrics], task_order: list[str]) -> dict[str, int]:
    mins: dict[str, int] = {}
    for task_id in task_order:
        task_metrics = [item.loc for item in metrics if item.task_id == task_id]
        mins[task_id] = min(task_metrics)
    return mins


def _metric_lookup(metrics: list[SnippetMetrics]) -> dict[tuple[str, str], SnippetMetrics]:
    return {(item.framework, item.task_id): item for item in metrics}


def _bool_tex(value: bool) -> str:
    return r"\cmark" if value else r"\xmark"


def _format_cell(metric: SnippetMetrics, min_loc: int) -> str:
    cell = f"{metric.loc} ({metric.imports})"
    if metric.loc == min_loc:
        return f"\\textbf{{{cell}}}"
    return cell


def make_main_table(metrics: list[SnippetMetrics], task_labels: dict[str, str], task_order: list[str]) -> str:
    framework_order = _framework_order(metrics)
    mins = _task_min_locs(metrics, task_order)
    lookup = _metric_lookup(metrics)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        (
            r"\caption{Accessibility proxies from canonical onboarding snippets for the three closest "
            r"pure-Python baselines considered in the paper. Entries report non-empty lines of code with "
            r"import counts in parentheses; lower is better. Guided tools denotes framework-distributed "
            r"onboarding support such as a guided command-line wizard or graphical problem builder.}"
        ),
        r"\label{tab:accessibility_proxies}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        (
            r"\textbf{Framework} & \textbf{Built-in Run} & \textbf{Custom Problem} & "
            r"\textbf{Expert Config} & \textbf{Guided Tools} \\"
        ),
        r"\midrule",
    ]

    for framework in framework_order:
        built_in = lookup[(framework, "built_in_run")]
        custom = lookup[(framework, "custom_problem")]
        expert = lookup[(framework, "expert_config")]
        lines.append(
            f"{framework} & "
            f"{_format_cell(built_in, mins['built_in_run'])} & "
            f"{_format_cell(custom, mins['custom_problem'])} & "
            f"{_format_cell(expert, mins['expert_config'])} & "
            f"{_bool_tex(built_in.guided_tools)} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{\textwidth}{\footnotesize Measurement rule: blank lines, comments, plotting, and output-export code are excluded.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_source(url: str, label: str) -> str:
    if url.startswith("local://"):
        escaped = url.removeprefix("local://").replace("_", r"\_")
        return rf"\texttt{{{escaped}}}"
    return rf"{label} (\url{{{url}}})"


def make_detail_table(metrics: list[SnippetMetrics], task_labels: dict[str, str], task_order: list[str]) -> str:
    framework_order = _framework_order(metrics)
    lookup = _metric_lookup(metrics)
    rows: list[SnippetMetrics] = []
    for task_id in task_order:
        for framework in framework_order:
            rows.append(lookup[(framework, task_id)])

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        (
            r"\caption{Detailed accessibility-proxy artifacts used in the manuscript. LOC counts non-empty, "
            r"non-comment lines in the canonical snippet; imports counts explicit \texttt{import}/\texttt{from} statements.}"
        ),
        r"\label{tab:accessibility_proxy_details}",
        r"\begin{tabular}{@{}llcccp{0.36\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Framework} & \textbf{LOC} & \textbf{Imports} & \textbf{Entry} & \textbf{Canonical source} \\",
        r"\midrule",
    ]

    current_task = None
    for row in rows:
        task_label = task_labels[row.task_id]
        if current_task is not None and current_task != row.task_id:
            lines.append(r"\midrule")
        current_task = row.task_id
        lines.append(
            f"{task_label} & {row.framework} & {row.loc} & {row.imports} & "
            f"\\texttt{{{row.entry_shape}}} & {_render_source(row.source_url, row.source_label)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def compile_tex(filename: str) -> None:
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename]
    subprocess.run(cmd, cwd=MANUSCRIPT_DIR, check=True)


def main() -> None:
    args = parse_args()
    task_labels, task_order, metrics = load_metrics(args.dataset)

    write_text(args.main_out, make_main_table(metrics, task_labels, task_order))
    write_text(args.details_out, make_detail_table(metrics, task_labels, task_order))

    print(f"Wrote: {args.main_out}")
    print(f"Wrote: {args.details_out}")

    if args.compile_main:
        compile_tex("main.tex")
    if args.compile_supplementary:
        compile_tex("supplementary.tex")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

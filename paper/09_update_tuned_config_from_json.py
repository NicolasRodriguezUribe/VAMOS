"""
VAMOS Paper Tuned-Config Update Script
=====================================
Generates a resolved NSGA-II configuration JSON from the racing tuner output and
updates the corresponding LaTeX table in paper/manuscript/main.tex.

Usage:
  python paper/09_update_tuned_config_from_json.py
  python paper/09_update_tuned_config_from_json.py --assignment-json experiments/tuned_nsgaii.json
  python paper/09_update_tuned_config_from_json.py --compile
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
MANUSCRIPT_DIR = Path(__file__).parent / "manuscript"

DEFAULT_ASSIGNMENT_JSON = ROOT_DIR / "experiments" / "tuned_nsgaii.json"
DEFAULT_RESOLVED_JSON = ROOT_DIR / "experiments" / "tuned_nsgaii_resolved.json"
DEFAULT_MAIN_TEX = MANUSCRIPT_DIR / "main.tex"

sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning import config_from_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update tuned-config table in main.tex from racing tuner JSON output.")
    parser.add_argument("--assignment-json", type=Path, default=DEFAULT_ASSIGNMENT_JSON, help="Path to tuned assignment JSON.")
    parser.add_argument("--resolved-json", type=Path, default=DEFAULT_RESOLVED_JSON, help="Path to write resolved config JSON.")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex.")
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with pdflatex after updating.")
    return parser.parse_args()


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _tt(text: str) -> str:
    return rf"\texttt{{{_tex_escape(text)}}}"


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return ""
    items: list[str] = []
    for k in sorted(params):
        items.append(f"{k}={_format_number(params[k])}")
    return ", ".join(items)


def _format_operator(op: tuple[str, dict[str, Any]] | list[Any]) -> str:
    name = str(op[0])
    params = op[1] if len(op) > 1 else {}
    params = {} if params is None else dict(params)
    params_txt = _format_params(params)
    if not params_txt:
        return _tt(name)
    return _tt(f"{name}({params_txt})")


def _external_archive_summary(cfg: NSGAIIConfig) -> str:
    archive_cfg = cfg.archive or {}
    size = int(archive_cfg.get("size", 0) or 0)
    if size <= 0:
        return "disabled"
    if bool(archive_cfg.get("unbounded", False)):
        return _tt(f"unbounded (size={size})")
    archive_type = cfg.archive_type or "crowding"
    return _tt(f"{archive_type} (size={size})")


def make_latex_config_table(cfg: NSGAIIConfig, *, caption: str, label: str) -> str:
    crossover = _format_operator(cfg.crossover)

    mut_name, mut_params = cfg.mutation
    mut_params = dict(mut_params or {})
    mut_prob_factor = cfg.mutation_prob_factor
    if mut_prob_factor is not None:
        mut_params = {**mut_params, "prob_factor": float(mut_prob_factor)}
    mutation = _tt(f"{mut_name}({_format_params(mut_params)})") if mut_params else _tt(str(mut_name))

    sel_name, sel_params = cfg.selection
    selection = _tt(f"{sel_name}({_format_params(dict(sel_params or {}))})")

    offspring_size = cfg.offspring_size if cfg.offspring_size is not None else cfg.pop_size

    rows = [
        (r"\texttt{pop\_size}", str(int(cfg.pop_size))),
        (r"\texttt{offspring\_size}", str(int(offspring_size))),
        ("Crossover", crossover),
        ("Mutation", mutation),
        ("Selection", selection),
        ("External archive", _external_archive_summary(cfg)),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l|p{0.68\linewidth}}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _find_table_bounds(content: str, label: str) -> tuple[int, int, str, str] | None:
    label_token = f"\\label{{{label}}}"
    label_pos = content.find(label_token)
    if label_pos == -1:
        return None

    begin_table = content.rfind(r"\begin{table}", 0, label_pos)
    begin_table_star = content.rfind(r"\begin{table*}", 0, label_pos)
    if begin_table_star > begin_table:
        begin_pos = begin_table_star
        begin_tag = r"\begin{table*}"
        end_tag = r"\end{table*}"
    else:
        begin_pos = begin_table
        begin_tag = r"\begin{table}"
        end_tag = r"\end{table}"

    if begin_pos == -1:
        return None

    end_pos = content.find(end_tag, label_pos)
    if end_pos == -1:
        return None
    end_pos += len(end_tag)
    return begin_pos, end_pos, begin_tag, end_tag


def _normalize_table_env(new_table: str, begin_tag: str, end_tag: str) -> str:
    if begin_tag.endswith("*") and "\\begin{table*}" not in new_table:
        new_table = new_table.replace(r"\begin{table}", begin_tag, 1)
        new_table = new_table.replace(r"\end{table}", end_tag, 1)
    return new_table


def replace_table_in_tex(content: str, label: str, new_table: str) -> tuple[str, bool]:
    bounds = _find_table_bounds(content, label)
    if not bounds:
        print(f"Warning: Table {label} not found")
        return content, False
    begin_pos, end_pos, begin_tag, end_tag = bounds
    new_table = _normalize_table_env(new_table, begin_tag, end_tag)
    return content[:begin_pos] + new_table + content[end_pos:], True


def compile_latex(tex_path: Path) -> bool:
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=tex_path.parent)
    return result.returncode == 0


def main() -> None:
    args = parse_args()

    if not args.main_tex.exists():
        raise SystemExit(f"main.tex not found: {args.main_tex}")

    if not args.assignment_json.exists():
        raise SystemExit(f"Assignment JSON not found: {args.assignment_json}")

    assignment = json.loads(args.assignment_json.read_text(encoding="utf-8"))
    if not isinstance(assignment, dict):
        raise SystemExit(f"Assignment JSON must be an object: {args.assignment_json}")

    cfg = config_from_assignment("nsgaii", assignment)
    if not isinstance(cfg, NSGAIIConfig):  # pragma: no cover
        raise SystemExit(f"Expected NSGAIIConfig, got: {type(cfg)}")

    args.resolved_json.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_json.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    caption = (
        r"Racing-tuned NSGA-II configuration used in the ablation study "
        r"(generated by \texttt{paper/09\_update\_tuned\_config\_from\_json.py})."
    )
    latex_table = make_latex_config_table(cfg, caption=caption, label="tab:racing_tuned_config")

    content = args.main_tex.read_text(encoding="utf-8")
    updated, changed = replace_table_in_tex(content, "tab:racing_tuned_config", latex_table)
    if not changed:
        raise SystemExit("Failed to update LaTeX table (label not found).")
    args.main_tex.write_text(updated, encoding="utf-8")

    if args.compile:
        ok = compile_latex(args.main_tex)
        if not ok:
            raise SystemExit("pdflatex failed.")


if __name__ == "__main__":
    main()


"""
VAMOS Paper Ablation Table Update Script
=======================================
Reads ablation CSV results and updates LaTeX tables in paper/manuscript/main.tex.

Usage:
  python paper/06_update_ablation_tables_from_csv.py
  python paper/06_update_ablation_tables_from_csv.py --csv experiments/ablation_aos_racing_tuner.csv
  python paper/06_update_ablation_tables_from_csv.py --empty
  python paper/06_update_ablation_tables_from_csv.py --compile
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
MANUSCRIPT_DIR = Path(__file__).parent / "manuscript"

DEFAULT_CSV = ROOT_DIR / "experiments" / "ablation_aos_racing_tuner.csv"
DEFAULT_MAIN_TEX = MANUSCRIPT_DIR / "main.tex"

_FAMILY_ORDER = ("ZDT", "DTLZ", "WFG")
_VARIANT_ORDER = ("baseline", "aos", "tuned")
_VARIANT_DISPLAY = {
    "baseline": "Baseline",
    "aos": "Baseline + AOS",
    "tuned": "Racing-tuned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update ablation tables in main.tex from ablation CSV results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to ablation CSV.")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex.")
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with pdflatex after updating.")
    parser.add_argument("--empty", action="store_true", help="Write placeholder tables (ignores CSV).")
    return parser.parse_args()


def get_family(problem_name: str) -> str:
    if problem_name.startswith("zdt"):
        return "ZDT"
    if problem_name.startswith("dtlz"):
        return "DTLZ"
    if problem_name.startswith("wfg"):
        return "WFG"
    return "Other"


def _format_cell(value: float | None, decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def make_placeholder_table(*, caption: str, label: str) -> str:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l|rrr|r}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{ZDT} & \textbf{DTLZ} & \textbf{WFG} & \textbf{Average} \\",
        r"\midrule",
        r"Baseline & -- & -- & -- & -- \\",
        r"Baseline + AOS & -- & -- & -- & -- \\",
        r"Racing-tuned & -- & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def make_family_table(
    family_df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    higher_is_better: bool,
    decimals: int,
) -> str:
    families = [c for c in _FAMILY_ORDER if c in family_df.columns]
    cols = [*families, "Average"]

    best_by_col: dict[str, float | None] = {}
    for col in cols:
        if col not in family_df.columns:
            best_by_col[col] = None
            continue
        series = family_df[col].dropna()
        if series.empty:
            best_by_col[col] = None
            continue
        best_by_col[col] = float(series.max() if higher_is_better else series.min())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l|" + "r" * len(families) + r"|r}",
        r"\toprule",
        r"\textbf{Variant} & "
        + " & ".join([rf"\textbf{{{f}}}" for f in families])
        + r" & \textbf{Average} \\",
        r"\midrule",
    ]

    for variant in _VARIANT_ORDER:
        if variant not in family_df.index:
            continue
        row = family_df.loc[variant]
        display = _VARIANT_DISPLAY.get(variant, variant)

        cells: list[str] = []
        for col in cols:
            v = None
            if col in family_df.columns:
                raw = row[col]
                if pd.notna(raw):
                    v = float(raw)
            cell = _format_cell(v, decimals)
            best = best_by_col.get(col)
            if v is not None and best is not None and abs(v - best) < 1e-12:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)

        lines.append(f"{display} & " + " & ".join(cells) + r" \\")

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
    csv_path: Path = args.csv
    main_tex: Path = args.main_tex

    if not main_tex.exists():
        raise SystemExit(f"main.tex not found: {main_tex}")

    runtime_caption = (
        r"Ablation study: median runtime (seconds) by problem family for \VAMOS{} variants "
        r"(generated by \texttt{paper/06\_update\_ablation\_tables\_from\_csv.py})."
    )
    hv_caption = (
        r"Ablation study: median normalized hypervolume by problem family for \VAMOS{} variants "
        r"(generated by \texttt{paper/06\_update\_ablation\_tables\_from\_csv.py})."
    )

    if args.empty or not csv_path.exists():
        if not csv_path.exists() and not args.empty:
            print(f"Warning: CSV not found, writing placeholders: {csv_path}")
        latex_runtime = make_placeholder_table(caption=runtime_caption, label="tab:ablation_runtime")
        latex_hv = make_placeholder_table(caption=hv_caption, label="tab:ablation_hypervolume")
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise SystemExit(f"No rows in CSV: {csv_path}")

        df["variant"] = df["variant"].astype(str).str.strip().str.lower()
        df["family"] = df["problem"].astype(str).str.strip().str.lower().apply(get_family)

        family_runtime = df.groupby(["variant", "family"])["runtime_seconds"].median().unstack()
        family_runtime["Average"] = family_runtime.mean(axis=1)
        family_runtime = family_runtime.reindex(index=list(_VARIANT_ORDER))

        family_hv = df.groupby(["variant", "family"])["hypervolume"].median().unstack()
        family_hv["Average"] = family_hv.mean(axis=1)
        family_hv = family_hv.reindex(index=list(_VARIANT_ORDER))

        latex_runtime = make_family_table(
            family_runtime,
            caption=runtime_caption,
            label="tab:ablation_runtime",
            higher_is_better=False,
            decimals=2,
        )
        latex_hv = make_family_table(
            family_hv,
            caption=hv_caption,
            label="tab:ablation_hypervolume",
            higher_is_better=True,
            decimals=3,
        )

    content = main_tex.read_text(encoding="utf-8")
    content, replaced_runtime = replace_table_in_tex(content, "tab:ablation_runtime", latex_runtime)
    content, replaced_hv = replace_table_in_tex(content, "tab:ablation_hypervolume", latex_hv)

    if not (replaced_runtime or replaced_hv):
        raise SystemExit("No tables updated (labels not found?)")

    main_tex.write_text(content, encoding="utf-8")
    print(f"Updated: {main_tex}")

    if args.compile:
        if compile_latex(main_tex):
            print(f"PDF compiled: {main_tex.parent / 'main.pdf'}")
        else:
            raise SystemExit("PDF compilation failed (pdflatex missing?)")


if __name__ == "__main__":
    main()

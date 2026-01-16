"""
VAMOS Paper Scaling Table Update Script
======================================
Reads scaling CSV results and updates LaTeX tables in paper/manuscript/main.tex.

Usage:
  python paper/07_update_scaling_tables_from_csv.py
  python paper/07_update_scaling_tables_from_csv.py --csv experiments/scaling_vectorization.csv
  python paper/07_update_scaling_tables_from_csv.py --empty
  python paper/07_update_scaling_tables_from_csv.py --compile
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
MANUSCRIPT_DIR = Path(__file__).parent / "manuscript"

DEFAULT_CSV = ROOT_DIR / "experiments" / "scaling_vectorization.csv"
DEFAULT_MAIN_TEX = MANUSCRIPT_DIR / "main.tex"

_POP_PROBLEM_ORDER = ("zdt4", "dtlz2", "wfg2")
_ENGINE_NUMPY = "numpy"
_ENGINE_NUMBA = "numba"
_JIT_PROBLEM_ORDER = ("zdt4", "dtlz2", "wfg2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update scaling tables in main.tex from scaling CSV results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to scaling CSV.")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex.")
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with pdflatex after updating.")
    parser.add_argument("--empty", action="store_true", help="Write placeholder tables (ignores CSV).")
    return parser.parse_args()


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


def make_placeholder_pop_table(*, caption: str, label: str) -> str:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r|rrr|r}",
        r"\toprule",
        r"\textbf{Pop.\ size} & \textbf{zdt4} & \textbf{dtlz2} & \textbf{wfg2} & \textbf{Average} \\",
        r"\midrule",
        r"50 & -- & -- & -- & -- \\",
        r"100 & -- & -- & -- & -- \\",
        r"200 & -- & -- & -- & -- \\",
        r"400 & -- & -- & -- & -- \\",
        r"800 & -- & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def make_placeholder_obj_table(*, caption: str, label: str) -> str:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r|rr|r}",
        r"\toprule",
        r"\textbf{$m$} & \textbf{$n$} & \textbf{Pop.\ size} & \textbf{Speedup} \\",
        r"\midrule",
        r"2 & -- & -- & -- \\",
        r"3 & -- & -- & -- \\",
        r"5 & -- & -- & -- \\",
        r"8 & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def make_placeholder_jit_table(*, caption: str, label: str) -> str:
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l|r}",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Median runtime (s)} \\",
        r"\midrule",
        r"zdt4 & -- \\",
        r"dtlz2 & -- \\",
        r"wfg2 & -- \\",
        r"\midrule",
        r"\textbf{Average} & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def _bold_if_best(cell: str, is_best: bool) -> str:
    return rf"\textbf{{{cell}}}" if is_best else cell


def make_pop_speedup_table(df_pop: pd.DataFrame, *, caption: str, label: str) -> str:
    df_pop = df_pop.copy()
    df_pop["engine"] = df_pop["engine"].astype(str).str.strip().str.lower()
    df_pop["problem"] = df_pop["problem"].astype(str).str.strip().str.lower()

    required = {"pop_size", "seed", "engine", "runtime_per_eval", "problem"}
    missing = required - set(df_pop.columns)
    if missing:
        raise ValueError(f"Missing columns in scaling CSV: {sorted(missing)}")

    pivot = df_pop.pivot_table(
        index=["pop_size", "problem", "seed"],
        columns="engine",
        values="runtime_per_eval",
        aggfunc="first",
    )
    pivot = pivot.dropna(subset=[_ENGINE_NUMPY, _ENGINE_NUMBA])
    pivot["speedup"] = pivot[_ENGINE_NUMPY] / pivot[_ENGINE_NUMBA]

    speedup = pivot.groupby(["pop_size", "problem"])["speedup"].median().unstack("problem")
    speedup = speedup.reindex(columns=list(_POP_PROBLEM_ORDER))
    speedup["Average"] = speedup.mean(axis=1)
    speedup = speedup.sort_index()

    best_by_col = {col: float(speedup[col].max()) for col in speedup.columns if col in speedup}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r|rrr|r}",
        r"\toprule",
        r"\textbf{Pop.\ size} & \textbf{zdt4} & \textbf{dtlz2} & \textbf{wfg2} & \textbf{Average} \\",
        r"\midrule",
    ]

    for pop_size in speedup.index.tolist():
        row = speedup.loc[pop_size]
        cells: list[str] = []
        for col in ["zdt4", "dtlz2", "wfg2", "Average"]:
            val = row.get(col)
            if pd.isna(val):
                cell = "--"
                is_best = False
            else:
                cell = f"{float(val):.2f}"
                is_best = abs(float(val) - best_by_col[col]) < 1e-12
            cells.append(_bold_if_best(cell, is_best))
        lines.append(f"{int(pop_size)} & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_obj_speedup_table(df_obj: pd.DataFrame, *, caption: str, label: str) -> str:
    df_obj = df_obj.copy()
    df_obj["engine"] = df_obj["engine"].astype(str).str.strip().str.lower()

    required = {"n_obj", "n_var", "pop_size", "seed", "engine", "runtime_per_eval"}
    missing = required - set(df_obj.columns)
    if missing:
        raise ValueError(f"Missing columns in scaling CSV: {sorted(missing)}")

    pivot = df_obj.pivot_table(
        index=["n_obj", "n_var", "pop_size", "seed"],
        columns="engine",
        values="runtime_per_eval",
        aggfunc="first",
    )
    pivot = pivot.dropna(subset=[_ENGINE_NUMPY, _ENGINE_NUMBA])
    pivot["speedup"] = pivot[_ENGINE_NUMPY] / pivot[_ENGINE_NUMBA]

    rows = pivot.groupby(["n_obj", "n_var", "pop_size"])["speedup"].median().reset_index()
    rows = rows.sort_values(["n_obj", "n_var", "pop_size"])

    best = float(rows["speedup"].max()) if not rows.empty else float("nan")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{r|rr|r}",
        r"\toprule",
        r"\textbf{$m$} & \textbf{$n$} & \textbf{Pop.\ size} & \textbf{Speedup} \\",
        r"\midrule",
    ]

    for _, row in rows.iterrows():
        m = int(row["n_obj"])
        n = int(row["n_var"])
        pop = int(row["pop_size"])
        sp = float(row["speedup"])
        cell = f"{sp:.2f}"
        lines.append(f"{m} & {n} & {pop} & {_bold_if_best(cell, abs(sp - best) < 1e-12)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_jit_runtime_table(df_jit: pd.DataFrame, *, timing_policy: str, caption: str, label: str) -> str:
    df_jit = df_jit.copy()
    df_jit["experiment"] = df_jit["experiment"].astype(str).str.strip().str.lower()
    df_jit["problem"] = df_jit["problem"].astype(str).str.strip().str.lower()
    df_jit["timing_policy"] = df_jit["timing_policy"].astype(str).str.strip().str.lower()

    required = {"problem", "seed", "timing_policy", "runtime_seconds"}
    missing = required - set(df_jit.columns)
    if missing:
        raise ValueError(f"Missing columns in scaling CSV: {sorted(missing)}")

    sub = df_jit[(df_jit["experiment"] == "jit_policy") & (df_jit["timing_policy"] == timing_policy)].copy()
    if sub.empty:
        raise ValueError(f"No rows for jit_policy/{timing_policy} in scaling CSV")

    med = sub.groupby("problem")["runtime_seconds"].median().reindex(list(_JIT_PROBLEM_ORDER))
    avg = float(med.mean()) if not med.empty else float("nan")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l|r}",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Median runtime (s)} \\",
        r"\midrule",
    ]

    for problem in _JIT_PROBLEM_ORDER:
        v = med.get(problem)
        cell = "--" if pd.isna(v) else f"{float(v):.2f}"
        lines.append(f"{problem} & {cell} \\\\")

    lines.append(r"\midrule")
    avg_cell = "--" if not math.isfinite(avg) else f"{avg:.2f}"
    lines.append(rf"\textbf{{Average}} & {avg_cell} \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv
    main_tex: Path = args.main_tex

    if not main_tex.exists():
        raise SystemExit(f"main.tex not found: {main_tex}")

    pop_caption = (
        r"Scaling study: speedup of \VAMOS{} (Numba) relative to \VAMOS{} (NumPy) as a function of population size "
        r"(median across seeds; higher is better). Generated by \texttt{paper/07\_update\_scaling\_tables\_from\_csv.py}."
    )
    obj_caption = (
        r"Scaling study: speedup of \VAMOS{} (Numba) relative to \VAMOS{} (NumPy) as a function of objectives on DTLZ2 "
        r"(canonical $n=m+9$; median across seeds; higher is better). Generated by \texttt{paper/07\_update\_scaling\_tables\_from\_csv.py}."
    )

    jit_warm_caption = (
        r"Numba timing policy (warm): median runtime (seconds) for representative problems, measured after a 2{,}000-evaluation "
        r"warmup run in the same process (30 seeds; $100{,}000$ evaluations per run). Generated by \texttt{paper/07\_update\_scaling\_tables\_from\_csv.py}."
    )
    jit_cold_caption = (
        r"Numba timing policy (cold): median runtime (seconds) for representative problems, measured from a fresh "
        r"process (includes JIT compilation; 30 seeds; $100{,}000$ evaluations per run). Generated by \texttt{paper/07\_update\_scaling\_tables\_from\_csv.py}."
    )

    if args.empty or not csv_path.exists():
        if not csv_path.exists() and not args.empty:
            print(f"Warning: CSV not found, writing placeholders: {csv_path}")
        latex_pop = make_placeholder_pop_table(caption=pop_caption, label="tab:scaling_pop_speedup")
        latex_obj = make_placeholder_obj_table(caption=obj_caption, label="tab:scaling_obj_speedup")
        latex_jit_warm = make_placeholder_jit_table(caption=jit_warm_caption, label="tab:numba_jit_warm")
        latex_jit_cold = make_placeholder_jit_table(caption=jit_cold_caption, label="tab:numba_jit_cold")
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise SystemExit(f"No rows in CSV: {csv_path}")

        df["experiment"] = df["experiment"].astype(str).str.strip().str.lower()
        df_pop = df[df["experiment"] == "population"].copy()
        df_obj = df[df["experiment"] == "objectives"].copy()
        df_jit = df[df["experiment"] == "jit_policy"].copy()

        latex_pop = make_pop_speedup_table(df_pop, caption=pop_caption, label="tab:scaling_pop_speedup")
        latex_obj = make_obj_speedup_table(df_obj, caption=obj_caption, label="tab:scaling_obj_speedup")

        if df_jit.empty:
            print("Warning: no jit_policy rows found; writing placeholder JIT timing tables")
            latex_jit_warm = make_placeholder_jit_table(caption=jit_warm_caption, label="tab:numba_jit_warm")
            latex_jit_cold = make_placeholder_jit_table(caption=jit_cold_caption, label="tab:numba_jit_cold")
        else:
            latex_jit_warm = make_jit_runtime_table(df_jit, timing_policy="warm", caption=jit_warm_caption, label="tab:numba_jit_warm")
            latex_jit_cold = make_jit_runtime_table(df_jit, timing_policy="cold", caption=jit_cold_caption, label="tab:numba_jit_cold")

    content = main_tex.read_text(encoding="utf-8")
    content, replaced_pop = replace_table_in_tex(content, "tab:scaling_pop_speedup", latex_pop)
    content, replaced_obj = replace_table_in_tex(content, "tab:scaling_obj_speedup", latex_obj)
    content, replaced_jit_warm = replace_table_in_tex(content, "tab:numba_jit_warm", latex_jit_warm)
    content, replaced_jit_cold = replace_table_in_tex(content, "tab:numba_jit_cold", latex_jit_cold)

    if not (replaced_pop or replaced_obj or replaced_jit_warm or replaced_jit_cold):
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

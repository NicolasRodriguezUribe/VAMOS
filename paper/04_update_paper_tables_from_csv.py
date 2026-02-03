"""
VAMOS Paper Table Update Script
===============================
Reads benchmark CSV results and updates LaTeX tables in main.tex.

Usage:
  python paper/04_update_paper_tables_from_csv.py
  python paper/04_update_paper_tables_from_csv.py --csv path\to\benchmark_paper.csv
  python paper/04_update_paper_tables_from_csv.py --compile
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
PAPER_DIR = Path(__file__).parent / "manuscript"
DATA_DIR = ROOT_DIR / "experiments"

DEFAULT_CSV = DATA_DIR / "benchmark_paper.csv"
DEFAULT_MAIN_TEX = PAPER_DIR / "main.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LaTeX tables in main.tex from benchmark CSV results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to benchmark CSV.")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex.")
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with pdflatex after updating.")
    parser.add_argument("--force", action="store_true", help="Write main.tex even if size check fails.")
    return parser.parse_args()


def get_family(problem_name: str) -> str:
    if problem_name.startswith("zdt"):
        return "ZDT"
    if problem_name.startswith("dtlz"):
        return "DTLZ"
    if problem_name.startswith("wfg"):
        return "WFG"
    return "Other"


def _backend_display_name(backend: str) -> str:
    return {"numpy": "NumPy", "numba": "Numba", "moocore": "MooCore"}.get(backend, backend)


def make_latex_table_a1(df_table: pd.DataFrame) -> str:
    """Table A.1: backends with row-wise minimum bolded."""
    # Requested order: Numba, Moocore, NumPy
    backend_order = ["numba", "moocore", "numpy"]

    # Check which exist
    valid_backends = [b for b in backend_order if b in df_table.columns]

    header_cols = [f"\\textbf{{{_backend_display_name(b)}}}" for b in valid_backends]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Detailed VAMOS backend comparison: median runtime (seconds) per problem.}",
        r"\label{tab:detailed_backends}",
        r"\begin{tabular}{l|" + "r" * len(valid_backends) + "}",
        r"\toprule",
        r"\textbf{Problem} & " + " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]

    for idx, row in df_table.iterrows():
        vals = {b: row[b] for b in valid_backends}
        # Filter NaNs for min calc
        numeric_vals = [v for v in vals.values() if pd.notna(v)]
        min_val = min(numeric_vals) if numeric_vals else -1

        row_str = []
        for col in valid_backends:
            v = vals[col]
            if pd.isna(v):
                row_str.append("-")
            elif abs(v - min_val) < 1e-9:
                row_str.append(f"\\textbf{{{v:.2f}}}")
            else:
                row_str.append(f"{v:.2f}")

        if idx == "Average":
            lines.append(r"\midrule")
            lines.append(f"\\textbf{{Average}} & {' & '.join(row_str)} \\\\")
        else:
            lines.append(f"{idx} & {' & '.join(row_str)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_latex_table_a2(df_table: pd.DataFrame) -> str:
    """Table A.2 (or A.6): detailed comparison of all frameworks."""
    # df_table indices are problems (or Average). Columns are frameworks.

    # Sort columns to have VAMOS variants first, then others
    # Explicit order requested: VAMOS, pymoo, DEAP, jMetalPy, Platypus
    cols = df_table.columns.tolist()

    order_map = {
        "VAMOS": 0,
        "pymoo": 1,
        "DEAP": 2,
        "jMetalPy": 3,
        "Platypus": 4,
        "VAMOS (numba)": 0,  # Handle alias if needed
    }

    def sort_key(c):
        base_c = c
        # Case insensitive check or direct map
        for k, v in order_map.items():
            if k.lower() == c.lower():
                return v
            if k in c:  # partial match
                return v
        return 99  # Others at end

    cols = sorted(cols, key=sort_key)

    # We might need to rotate table or ensure it fits.
    # If too many columns, maybe small font?
    # For now, standard table.

    lines = [
        r"\begin{table*}[htbp]",  # Use table* for wide content
        r"\tiny",  # Reduce font size to fit all columns
        r"\centering",
        r"\caption{Detailed comparison of median runtime (seconds) across all frameworks.}",
        r"\label{tab:detailed_comparison}",
        r"\begin{tabular}{l|" + "r" * len(cols) + "}",
        r"\toprule",
        r"\textbf{Problem} & " + " & ".join([f"\\textbf{{{c}}}" for c in cols]) + r" \\",
        r"\midrule",
    ]

    for idx, row in df_table.iterrows():
        row_str = []

        # Find min value for bolding
        # Filter out NaNs if any, though median shouldn't have them if data exists
        valid_vals = [row[c] for c in cols if pd.notna(row[c])]
        min_val = min(valid_vals) if valid_vals else -1

        for col in cols:
            val = row[col]
            if pd.isna(val):
                row_str.append("-")
            elif val == min_val:
                row_str.append(f"\\textbf{{{val:.2f}}}")
            else:
                row_str.append(f"{val:.2f}")

        if idx == "Average":
            lines.append(r"\midrule")
            lines.append(f"\\textbf{{Average}} & {' & '.join(row_str)} \\\\")
        else:
            lines.append(f"{idx} & {' & '.join(row_str)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def make_latex_table_3(vamos_fam: pd.DataFrame) -> str:
    """Table 3: VAMOS backend comparison by family (ZDT, DTLZ, WFG)."""
    families = [col for col in ["ZDT", "DTLZ", "WFG"] if col in vamos_fam.columns]
    col_spec = "l|" + "r" * len(families) + "|r"
    header_cols = " & ".join([f"\\textbf{{{f}}}" for f in families]) + " & \\textbf{Average}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{VAMOS backend comparison: median runtime (seconds) by problem family.}",
        r"\label{tab:backends}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{Backend}} & {header_cols} \\\\",
        r"\midrule",
    ]

    # Desired backend order: Numba, Moocore, NumPy
    backend_order = ["numba", "moocore", "numpy"]
    valid_backends = [b for b in backend_order if b in vamos_fam.index]

    # Calculate min per column for bolding (across backends)
    min_vals = {}
    for col in families + ["Average"]:
        # Get values for this column across all valid backends
        col_vals = vamos_fam.loc[valid_backends, col]
        min_vals[col] = col_vals.min()

    for backend in valid_backends:
        row = vamos_fam.loc[backend]
        vals = {f: row[f] for f in families}
        vals["Average"] = row["Average"]

        row_str = []
        # Iterate families + Average
        for col in families + ["Average"]:
            v = vals[col]
            # Bold if it matches the column minimum
            if abs(v - min_vals[col]) < 1e-9:
                row_str.append(f"\\textbf{{{v:.2f}}}")
            else:
                row_str.append(f"{v:.2f}")

        lines.append(f"{_backend_display_name(backend)} & {' & '.join(row_str)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_latex_table_4(family_df: pd.DataFrame) -> str:
    """Table 4: Framework comparison by family."""
    # family_df index = Frameworks, cols = ZDT, DTLZ, WFG, Average

    # Sort frameworks
    idx_list = sorted(family_df.index.tolist())

    order_map = {"VAMOS": 0, "pymoo": 1, "DEAP": 2, "jMetalPy": 3, "Platypus": 4}

    def sort_key(c):
        for k, v in order_map.items():
            if k.lower() == c.lower():
                return v
            if k in c:
                return v
        return 99

    idx_list = sorted(idx_list, key=sort_key)

    families = [col for col in ["ZDT", "DTLZ", "WFG"] if col in family_df.columns]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Median runtime (seconds) by problem family across all frameworks.}",
        r"\label{tab:frameworks_perf}",
        r"\begin{tabular}{l|" + "r" * len(families) + "|r}",
        r"\toprule",
    ]

    header = " & ".join([f"\\textbf{{{f}}}" for f in families]) + " & \\textbf{Average}"
    lines.append(f"\\textbf{{Framework}} & {header} \\\\")
    lines.append(r"\midrule")

    for fw in idx_list:
        row = family_df.loc[fw]

        # Determine if this row has the best average? Or bolding per column?
        # Let's bold the best per column

        row_str = []
        for col in families + ["Average"]:
            val = row[col]
            # Check if this is the min in the column
            col_min = family_df[col].min()
            if abs(val - col_min) < 1e-9:
                row_str.append(f"\\textbf{{{val:.2f}}}")
            else:
                row_str.append(f"{val:.2f}")

        lines.append(f"{fw} & {' & '.join(row_str)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_latex_table_hv_summary(median_df: pd.DataFrame, iqr_df: pd.DataFrame) -> str:
    """Table 10: Normalized hypervolume summary (median (IQR)) by family.

    We compute per-problem median(HV) across seeds, then summarize these per-problem medians
    using the median and interquartile range (IQR) across problems within each family.
    """
    families = [col for col in ["ZDT", "DTLZ", "WFG"] if col in median_df.columns]
    cols = families + ["Overall"]

    order_map = {"VAMOS": 0, "pymoo": 1, "DEAP": 2, "jMetalPy": 3, "Platypus": 4}

    def sort_key(c: str) -> int:
        for k, v in order_map.items():
            if k.lower() == c.lower():
                return v
            if k in c:
                return v
        return 99

    idx_list = sorted(median_df.index.tolist(), key=sort_key)

    # Bold best (max) median per column, but use the displayed precision to avoid
    # confusing cases where values tie after rounding.
    best_median_rounded = {
        col: round(float(median_df[col].max()), 3) for col in cols if col in median_df.columns
    }

    def fmt_cell(med: float, iqr: float, bold: bool) -> str:
        s = f"{med:.3f} ({iqr:.3f})"
        return f"\\textbf{{{s}}}" if bold else s

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Normalized hypervolume summary (median (IQR)) by problem family across frameworks.}",
        r"\label{tab:frameworks_hv}",
        r"\begin{tabular}{l|" + "r" * len(families) + "|r}",
        r"\toprule",
    ]

    header = " & ".join([f"\\textbf{{{c}}}" for c in families]) + " & \\textbf{Overall}"
    lines.append(f"\\textbf{{Framework}} & {header} \\\\")
    lines.append(r"\midrule")

    for fw in idx_list:
        row_parts: list[str] = []
        for col in cols:
            med = median_df.at[fw, col] if col in median_df.columns else float("nan")
            iqr = iqr_df.at[fw, col] if col in iqr_df.columns else float("nan")
            if pd.isna(med) or pd.isna(iqr):
                row_parts.append("-")
                continue
            bold = round(float(med), 3) == best_median_rounded[col]
            row_parts.append(fmt_cell(float(med), float(iqr), bold=bold))

        # Insert column separator before Overall
        lines.append(f"{fw} & {' & '.join(row_parts[:-1])} & {row_parts[-1]} \\\\")

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
    """Replace a LaTeX table by label without spanning multiple tables."""
    if not new_table:
        print(f"Skipping {label}: no table content")
        return content, False

    bounds = _find_table_bounds(content, label)
    if not bounds:
        print(f"Warning: Table {label} not found")
        return content, False

    begin_pos, end_pos, begin_tag, end_tag = bounds
    new_table = _normalize_table_env(new_table, begin_tag, end_tag)
    return content[:begin_pos] + new_table + content[end_pos:], True


def compile_latex(tex_path: Path) -> bool:
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
    )
    for ext in [".aux", ".log", ".out"]:
        aux = tex_path.parent / (tex_path.stem + ext)
        try:
            aux.unlink()
        except OSError:
            pass
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv
    main_tex: Path = args.main_tex

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not main_tex.exists():
        raise SystemExit(f"main.tex not found: {main_tex}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No rows in CSV: {csv_path}")

    df["family"] = df["problem"].apply(get_family)

    family = df.groupby(["framework", "family"])["runtime_seconds"].median().unstack()
    family["Average"] = family.mean(axis=1)

    vamos_family = family.loc[family.index.str.contains("VAMOS")].copy()
    vamos_family.index = vamos_family.index.str.replace("VAMOS (", "", regex=False).str.replace(")", "", regex=False)

    detail = df.groupby(["framework", "problem"])["runtime_seconds"].median().unstack()

    backends_detail = detail.loc[detail.index.str.contains("VAMOS")].T.copy()
    backends_detail.columns = backends_detail.columns.str.replace("VAMOS (", "", regex=False).str.replace(")", "", regex=False)
    avg_row = backends_detail.mean()
    avg_row.name = "Average"
    backends_detail = pd.concat([backends_detail, avg_row.to_frame().T])

    # --- TABLE 4 Data Preparation ---
    # User Request: Only show "VAMOS" (using Numba data) vs others.
    # Exclude VAMOS (numpy), VAMOS (moocore).

    # Filter family df
    family_filtered = family.copy()
    if "VAMOS (numba)" in family_filtered.index:
        family_filtered = family_filtered.rename(index={"VAMOS (numba)": "VAMOS"})

    # Remove other VAMOS rows
    family_filtered = family_filtered.loc[~family_filtered.index.str.contains(r"VAMOS \(")]

    # --- TABLE 10 Data Preparation (Normalized HV summary) ---
    # We summarize solution quality without hypothesis tests: per-problem median(HV) across seeds,
    # then median and IQR of these per-problem medians within each family.
    df_hv = df.copy()
    if "VAMOS (numba)" in set(df_hv["framework"]):
        df_hv.loc[df_hv["framework"] == "VAMOS (numba)", "framework"] = "VAMOS"
    df_hv = df_hv.loc[~df_hv["framework"].str.startswith("VAMOS (")]

    hv_problem = (
        df_hv.groupby(["framework", "family", "problem"])["hypervolume"]
        .median()
        .reset_index()
    )

    hv_median = hv_problem.groupby(["framework", "family"])["hypervolume"].median().unstack()
    hv_q1 = hv_problem.groupby(["framework", "family"])["hypervolume"].quantile(0.25).unstack()
    hv_q3 = hv_problem.groupby(["framework", "family"])["hypervolume"].quantile(0.75).unstack()
    hv_iqr = hv_q3 - hv_q1

    overall_median = hv_problem.groupby("framework")["hypervolume"].median()
    overall_iqr = hv_problem.groupby("framework")["hypervolume"].quantile(0.75) - hv_problem.groupby("framework")[
        "hypervolume"
    ].quantile(0.25)

    hv_median["Overall"] = overall_median
    hv_iqr["Overall"] = overall_iqr

    # --- TABLE A.2 (Comparison) Data Preparation ---
    comparison_detail = detail.T.copy()

    # Rename columns and filter
    if "VAMOS (numba)" in comparison_detail.columns:
        comparison_detail = comparison_detail.rename(columns={"VAMOS (numba)": "VAMOS"})

    # Drop other VAMOS columns
    cols_to_drop = [c for c in comparison_detail.columns if c.startswith("VAMOS (")]
    comparison_detail = comparison_detail.drop(columns=cols_to_drop)

    # Calculate Average row for the filtered table
    avg_row = comparison_detail.mean()
    avg_row.name = "Average"
    comparison_detail = pd.concat([comparison_detail, avg_row.to_frame().T])

    expected_backends = ["numpy", "moocore", "numba"]
    missing_backends = [b for b in expected_backends if b not in backends_detail.columns]
    if missing_backends:
        print(f"Warning: missing backend columns for Table A.1: {missing_backends}")

    table_3_latex = make_latex_table_3(vamos_family) if not vamos_family.empty else ""
    table_4_latex = make_latex_table_4(family_filtered) if not family_filtered.empty else ""
    table_10_latex = make_latex_table_hv_summary(hv_median, hv_iqr) if not hv_median.empty else ""
    table_a1_latex = make_latex_table_a1(backends_detail) if not missing_backends else ""
    table_a2_latex = make_latex_table_a2(comparison_detail) if not comparison_detail.empty else ""

    print("\n" + "=" * 60)
    print("TABLE 3 - Backend by Family")
    print("=" * 60)
    print(table_3_latex)
    print()
    print("=" * 60)
    print("TABLE 4 - Frameworks by Family")
    print("=" * 60)
    print(table_4_latex)
    print()
    print("=" * 60)
    print("TABLE 10 - Normalized HV Summary")
    print("=" * 60)
    print(table_10_latex)
    print()
    print("=" * 60)
    print("TABLE A.1 - Backends")
    print("=" * 60)
    print(table_a1_latex)
    print()
    print("=" * 60)
    print("TABLE A.2 - Comparison")
    print("=" * 60)
    print(table_a2_latex)

    content = main_tex.read_text(encoding="utf-8")
    original_len = len(content)

    replaced_any = False
    for label, table in [
        ("tab:backends", table_3_latex),
        ("tab:frameworks_perf", table_4_latex),
        ("tab:frameworks_hv", table_10_latex),
        ("tab:detailed_backends", table_a1_latex),
        ("tab:detailed_comparison", table_a2_latex),
    ]:
        content, replaced = replace_table_in_tex(content, label, table)
        replaced_any = replaced_any or replaced

    if not replaced_any:
        print("No tables were updated.")
        return

    if not args.force and len(content) < original_len * 0.9:
        print("ERROR: Content too short, skipping write (use --force to override)")
        return

    main_tex.write_text(content, encoding="utf-8")
    print(f"\nmain.tex updated ({len(content)} bytes)")

    if args.compile:
        if compile_latex(main_tex):
            print(f"PDF compiled: {main_tex.parent / 'main.pdf'}")
        else:
            print("PDF compilation failed (pdflatex may not be installed)")


if __name__ == "__main__":
    main()

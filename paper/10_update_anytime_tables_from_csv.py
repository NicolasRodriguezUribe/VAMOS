"""
VAMOS Paper Anytime Ablation Table Update Script
===============================================
Reads anytime (checkpoint) ablation CSV results and updates LaTeX tables in
paper/manuscript/main.tex.

This script summarizes:
  - AUC(HV vs evals) using checkpoint HV values
  - Time-to-target: earliest checkpoint reaching 0.99× baseline-final HV
    (per problem), plus success rate.

Usage:
  python paper/10_update_anytime_tables_from_csv.py
  python paper/10_update_anytime_tables_from_csv.py --csv experiments/ablation_aos_anytime.csv
  python paper/10_update_anytime_tables_from_csv.py --empty
  python paper/10_update_anytime_tables_from_csv.py --compile
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
MANUSCRIPT_DIR = Path(__file__).parent / "manuscript"

DEFAULT_CSV = ROOT_DIR / "experiments" / "ablation_aos_anytime.csv"
DEFAULT_MAIN_TEX = MANUSCRIPT_DIR / "main.tex"

_FAMILY_ORDER = ("ZDT", "DTLZ", "WFG")
_VARIANT_ORDER = ("baseline", "aos", "tuned", "tuned_aos")
_VARIANT_DISPLAY = {
    "baseline": "Baseline",
    "aos": "Baseline + AOS",
    "tuned": "Racing-tuned",
    "tuned_aos": "Racing-tuned + AOS",
}

_TTT_FACTOR = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update anytime ablation tables in main.tex from checkpoint CSV results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to anytime (checkpoint) ablation CSV.")
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX, help="Path to main.tex.")
    parser.add_argument(
        "--n-evals",
        type=int,
        default=0,
        help="Filter to this evaluation budget (default: auto-select max n_evals in CSV).",
    )
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with pdflatex after updating.")
    parser.add_argument("--empty", action="store_true", help="Write placeholder tables (ignores CSV).")
    return parser.parse_args()


def get_family(problem_name: str) -> str:
    name = str(problem_name).strip().lower()
    if name.startswith("zdt"):
        return "ZDT"
    if name.startswith("dtlz"):
        return "DTLZ"
    if name.startswith("wfg"):
        return "WFG"
    return "Other"


def _format_cell(value: float | None, decimals: int) -> str:
    if value is None:
        return "--"
    if decimals <= 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def make_placeholder_table(*, caption: str, label: str, decimals: int = 3) -> str:
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
        r"Racing-tuned + AOS & -- & -- & -- & -- \\",
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


def _compute_auc(hv: np.ndarray, evals: np.ndarray, *, max_evals: int) -> float:
    hv = np.asarray(hv, dtype=float)
    evals = np.asarray(evals, dtype=float)
    if hv.size == 0 or evals.size == 0:
        return float("nan")
    order = np.argsort(evals)
    hv = hv[order]
    evals = evals[order]
    if evals[0] > 0:
        hv = np.concatenate([[0.0], hv])
        evals = np.concatenate([[0.0], evals])
    if max_evals <= 0:
        max_evals = int(evals.max(initial=0))
    area = float(np.trapezoid(hv, evals)) if hasattr(np, "trapezoid") else float(np.trapz(hv, evals))
    return area / float(max_evals) if max_evals > 0 else float("nan")


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv
    main_tex: Path = args.main_tex

    if not main_tex.exists():
        raise SystemExit(f"main.tex not found: {main_tex}")

    auc_caption = (
        r"Anytime ablation: median AUC of normalized hypervolume over evaluations "
        r"(HV checkpoints integrated over the evaluation budget; generated by \texttt{paper/10\_update\_anytime\_tables\_from\_csv.py})."
    )
    ttt_evals_caption = (
        r"Anytime ablation: median evaluations to reach $0.99\times$ the baseline-final HV target (per problem), "
        r"aggregated by family; generated by \texttt{paper/10\_update\_anytime\_tables\_from\_csv.py}."
    )
    ttt_success_caption = (
        r"Anytime ablation: success rate (\%) reaching the $0.99\times$ baseline-final HV target by the full budget, "
        r"aggregated by family; generated by \texttt{paper/10\_update\_anytime\_tables\_from\_csv.py}."
    )

    if args.empty or not csv_path.exists():
        if not csv_path.exists() and not args.empty:
            print(f"Warning: CSV not found, writing placeholders: {csv_path}")
        latex_auc = make_placeholder_table(caption=auc_caption, label="tab:anytime_auc", decimals=3)
        latex_ttt_evals = make_placeholder_table(caption=ttt_evals_caption, label="tab:anytime_ttt_evals", decimals=0)
        latex_ttt_success = make_placeholder_table(caption=ttt_success_caption, label="tab:anytime_ttt_success", decimals=1)
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise SystemExit(f"No rows in CSV: {csv_path}")

        required = {"variant", "problem", "seed", "evals", "runtime_seconds", "hypervolume", "n_evals"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

        df["variant"] = df["variant"].astype(str).str.strip().str.lower()
        df["problem"] = df["problem"].astype(str).str.strip().str.lower()
        df["family"] = df["problem"].apply(get_family)
        df["seed"] = df["seed"].astype(int)
        df["evals"] = df["evals"].astype(int)
        df["n_evals"] = df["n_evals"].astype(int)

        available = sorted(int(x) for x in df["n_evals"].dropna().unique())
        if not available:
            raise SystemExit("No n_evals values found in CSV.")
        target = int(args.n_evals) if int(args.n_evals) > 0 else int(max(available))
        if target not in available:
            raise SystemExit(f"Requested --n-evals={target} not present in CSV (available: {available}).")
        if len(available) > 1:
            print(f"Filtering anytime CSV to n_evals={target} (available: {available})")
        df = df[df["n_evals"] == target].copy()

        # Guard against mismatched checkpoints (e.g., evals > n_evals) when merging runs.
        df = df[df["evals"] <= df["n_evals"]].copy()

        max_evals = int(target)

        # ----------------------------
        # AUC per (variant,problem,seed)
        # ----------------------------
        auc_rows = []
        for (variant, problem, seed), g in df.groupby(["variant", "problem", "seed"], sort=False):
            g = g.sort_values("evals")
            auc = _compute_auc(g["hypervolume"].to_numpy(), g["evals"].to_numpy(), max_evals=max_evals)
            auc_rows.append(
                {
                    "variant": variant,
                    "problem": problem,
                    "seed": int(seed),
                    "family": get_family(problem),
                    "auc": float(auc),
                }
            )
        df_auc = pd.DataFrame(auc_rows)

        family_auc = df_auc.groupby(["variant", "family"])["auc"].median().unstack()
        family_auc["Average"] = family_auc.mean(axis=1)
        family_auc = family_auc.reindex(index=list(_VARIANT_ORDER))

        latex_auc = make_family_table(
            family_auc,
            caption=auc_caption,
            label="tab:anytime_auc",
            higher_is_better=True,
            decimals=3,
        )

        # ----------------------------
        # Time-to-target (relative to baseline final HV)
        # ----------------------------
        base = df[df["variant"] == "baseline"].copy()
        base_final = base[base["evals"] == base["n_evals"]].copy()
        if base_final.empty:
            raise SystemExit("No baseline final checkpoint rows found (evals == n_evals).")

        targets = base_final.groupby("problem")["hypervolume"].median() * float(_TTT_FACTOR)
        targets = targets.to_dict()

        ttt_rows = []
        for (variant, problem, seed), g in df.groupby(["variant", "problem", "seed"], sort=False):
            target = targets.get(problem)
            if target is None:
                continue
            g = g.sort_values("evals")
            reached = g[g["hypervolume"] >= float(target)]
            if reached.empty:
                ttt_rows.append(
                    {
                        "variant": variant,
                        "problem": problem,
                        "seed": int(seed),
                        "family": get_family(problem),
                        "success": 0,
                        "evals_to_target": np.nan,
                        "seconds_to_target": np.nan,
                    }
                )
                continue
            first = reached.iloc[0]
            ttt_rows.append(
                {
                    "variant": variant,
                    "problem": problem,
                    "seed": int(seed),
                    "family": get_family(problem),
                    "success": 1,
                    "evals_to_target": float(first["evals"]),
                    "seconds_to_target": float(first["runtime_seconds"]),
                }
            )

        df_ttt = pd.DataFrame(ttt_rows)
        if df_ttt.empty:
            raise SystemExit("No time-to-target rows computed (missing targets?)")

        family_ttt_evals = (
            df_ttt[df_ttt["success"] == 1].groupby(["variant", "family"])["evals_to_target"].median().unstack()
        )
        family_ttt_evals["Average"] = family_ttt_evals.mean(axis=1)
        family_ttt_evals = family_ttt_evals.reindex(index=list(_VARIANT_ORDER))

        family_ttt_success = df_ttt.groupby(["variant", "family"])["success"].mean().unstack() * 100.0
        family_ttt_success["Average"] = family_ttt_success.mean(axis=1)
        family_ttt_success = family_ttt_success.reindex(index=list(_VARIANT_ORDER))

        latex_ttt_evals = make_family_table(
            family_ttt_evals,
            caption=ttt_evals_caption,
            label="tab:anytime_ttt_evals",
            higher_is_better=False,
            decimals=0,
        )
        latex_ttt_success = make_family_table(
            family_ttt_success,
            caption=ttt_success_caption,
            label="tab:anytime_ttt_success",
            higher_is_better=True,
            decimals=1,
        )

    content = main_tex.read_text(encoding="utf-8")
    content, replaced_auc = replace_table_in_tex(content, "tab:anytime_auc", latex_auc)
    content, replaced_ttt_evals = replace_table_in_tex(content, "tab:anytime_ttt_evals", latex_ttt_evals)
    content, replaced_ttt_success = replace_table_in_tex(content, "tab:anytime_ttt_success", latex_ttt_success)

    if not (replaced_auc or replaced_ttt_evals or replaced_ttt_success):
        raise SystemExit("No anytime tables updated (labels not found?)")

    main_tex.write_text(content, encoding="utf-8")
    print(f"Updated: {main_tex}")

    if args.compile:
        if compile_latex(main_tex):
            print(f"PDF compiled: {main_tex.parent / 'main.pdf'}")
        else:
            raise SystemExit("PDF compilation failed (pdflatex missing?)")


if __name__ == "__main__":
    main()

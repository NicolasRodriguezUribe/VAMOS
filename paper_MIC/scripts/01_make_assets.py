"""
Generate MIC paper assets (tables + figures) from the committed experiment CSVs.

Usage (recommended):
  .\\.venv\\Scripts\\python.exe paper_MIC\\scripts\\01_make_assets.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT_DIR / "paper_MIC"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"


_FAMILY_ORDER = ("ZDT", "DTLZ", "WFG")
_VARIANT_ORDER = ("baseline", "aos")
_VARIANT_DISPLAY = {"baseline": "Baseline", "aos": "Baseline + AOS"}
_PROBLEM_ORDER = (
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt6",
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz4",
    "dtlz5",
    "dtlz6",
    "dtlz7",
    "wfg1",
    "wfg2",
    "wfg3",
    "wfg4",
    "wfg5",
    "wfg6",
    "wfg7",
    "wfg8",
    "wfg9",
)


def _family(problem_name: str) -> str:
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
    return f"{value:.{decimals}f}"


def _bold_if_best(value: float | None, best: float | None, *, eps: float = 1e-12) -> str:
    if value is None:
        return "--"
    text = str(value)
    if best is not None and abs(value - best) < eps:
        return rf"\textbf{{{text}}}"
    return text


def _make_family_table(
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
        r"\begin{table}[t]",
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
            v: float | None = None
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


def _make_per_problem_table(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    value_col: str,
    higher_is_better: bool,
    decimals: int,
) -> str:
    pivot = df.pivot(index="problem", columns="variant", values=value_col).reindex(index=list(_PROBLEM_ORDER))
    pivot = pivot.dropna(how="all")
    pivot["delta"] = pivot.get("aos") - pivot.get("baseline")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l|rrr}",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Baseline} & \textbf{AOS} & $\Delta$ \\",
        r"\midrule",
    ]

    last_family = None
    for problem, row in pivot.iterrows():
        fam = _family(problem)
        if last_family is not None and fam != last_family:
            lines.append(r"\midrule")
        last_family = fam

        base = None if pd.isna(row.get("baseline")) else float(row.get("baseline"))
        aos = None if pd.isna(row.get("aos")) else float(row.get("aos"))
        delta = None if pd.isna(row.get("delta")) else float(row.get("delta"))

        base_cell = _format_cell(base, decimals)
        aos_cell = _format_cell(aos, decimals)
        delta_cell = _format_cell(delta, decimals)

        if base is not None and aos is not None:
            best = max(base, aos) if higher_is_better else min(base, aos)
            if abs(base - best) < 1e-12:
                base_cell = rf"\textbf{{{base_cell}}}"
            if abs(aos - best) < 1e-12:
                aos_cell = rf"\textbf{{{aos_cell}}}"

        lines.append(f"{problem} & {base_cell} & {aos_cell} & {delta_cell} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def plot_delta_bars(
    per_problem: pd.DataFrame,
    *,
    value_col: str,
    out_pdf: Path,
    title: str,
    ylabel: str,
) -> None:
    df = per_problem.pivot(index="problem", columns="variant", values=value_col).copy()
    df = df.reindex(index=list(_PROBLEM_ORDER)).dropna(how="all")
    df["delta"] = df.get("aos") - df.get("baseline")
    df = df.sort_values("delta", ascending=False)

    x = np.arange(df.shape[0])
    y = df["delta"].to_numpy(dtype=float)
    colors = np.where(y >= 0.0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 3.1))
    ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in df.index], rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class AblationSummary:
    hv_family: pd.DataFrame
    runtime_family: pd.DataFrame
    hv_per_problem: pd.DataFrame
    runtime_per_problem: pd.DataFrame
    hv_overall_median: dict[str, float]
    runtime_overall_median: dict[str, float]
    n_problems: int
    n_seeds: int
    wins: int
    losses: int
    best_problem: str
    best_delta: float
    worst_problem: str
    worst_delta: float


def summarize_ablation(csv_path: Path, *, n_evals: int = 0) -> AblationSummary:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows in CSV: {csv_path}")

    if "n_evals" in df.columns:
        df["n_evals"] = df["n_evals"].astype(int)
        available = sorted(int(x) for x in df["n_evals"].dropna().unique())
        if not available:
            raise ValueError("CSV contains n_evals but no values.")
        target = int(n_evals) if int(n_evals) > 0 else int(max(available))
        if target not in available:
            raise ValueError(f"Requested n_evals={target} not present (available: {available}).")
        df = df[df["n_evals"] == target].copy()

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"].isin(_VARIANT_ORDER)].copy()

    df["family"] = df["problem"].astype(str).map(_family)
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)

    hv_family = df.groupby(["variant", "family"])["hypervolume"].median().unstack()
    hv_family["Average"] = hv_family.mean(axis=1)
    hv_family = hv_family.reindex(index=list(_VARIANT_ORDER))

    runtime_family = df.groupby(["variant", "family"])["runtime_seconds"].median().unstack()
    runtime_family["Average"] = runtime_family.mean(axis=1)
    runtime_family = runtime_family.reindex(index=list(_VARIANT_ORDER))

    hv_overall_median = df.groupby("variant")["hypervolume"].median().to_dict()
    runtime_overall_median = df.groupby("variant")["runtime_seconds"].median().to_dict()

    hv_per_problem = df.groupby(["variant", "problem"], as_index=False)["hypervolume"].median()
    runtime_per_problem = df.groupby(["variant", "problem"], as_index=False)["runtime_seconds"].median()

    hv_wide = hv_per_problem.pivot(index="problem", columns="variant", values="hypervolume")
    hv_delta = (hv_wide.get("aos") - hv_wide.get("baseline")).rename("delta")
    wins = int((hv_delta > 0).sum())
    losses = int((hv_delta < 0).sum())

    best_problem = str(hv_delta.idxmax())
    best_delta = float(hv_delta.max())
    worst_problem = str(hv_delta.idxmin())
    worst_delta = float(hv_delta.min())

    return AblationSummary(
        hv_family=hv_family,
        runtime_family=runtime_family,
        hv_per_problem=hv_per_problem,
        runtime_per_problem=runtime_per_problem,
        hv_overall_median={str(k): float(v) for k, v in hv_overall_median.items()},
        runtime_overall_median={str(k): float(v) for k, v in runtime_overall_median.items()},
        n_problems=int(hv_wide.shape[0]),
        n_seeds=int(df["seed"].nunique()),
        wins=wins,
        losses=losses,
        best_problem=best_problem,
        best_delta=best_delta,
        worst_problem=worst_problem,
        worst_delta=worst_delta,
    )


def write_summary_macros(summary: AblationSummary, out_path: Path) -> None:
    hv_base = float(summary.hv_overall_median["baseline"])
    hv_aos = float(summary.hv_overall_median["aos"])
    rt_base = float(summary.runtime_overall_median["baseline"])
    rt_aos = float(summary.runtime_overall_median["aos"])
    rt_overhead_pct = 100.0 * (rt_aos - rt_base) / max(rt_base, 1e-12)

    lines = [
        "% Auto-generated by paper_MIC/scripts/01_make_assets.py",
        rf"\newcommand{{\AOSNProblems}}{{{summary.n_problems}}}",
        rf"\newcommand{{\AOSNSeeds}}{{{summary.n_seeds}}}",
        rf"\newcommand{{\AOSWins}}{{{summary.wins}}}",
        rf"\newcommand{{\AOSLosses}}{{{summary.losses}}}",
        rf"\newcommand{{\AOSHVMedianBaseline}}{{{hv_base:.3f}}}",
        rf"\newcommand{{\AOSHVMedianAOS}}{{{hv_aos:.3f}}}",
        rf"\newcommand{{\AOSRuntimeMedianBaseline}}{{{rt_base:.2f}}}",
        rf"\newcommand{{\AOSRuntimeMedianAOS}}{{{rt_aos:.2f}}}",
        rf"\newcommand{{\AOSRuntimeOverheadPct}}{{{rt_overhead_pct:.0f}}}",
        rf"\newcommand{{\AOSBestProblem}}{{{summary.best_problem}}}",
        rf"\newcommand{{\AOSBestDelta}}{{{summary.best_delta:.3f}}}",
        rf"\newcommand{{\AOSWorstProblem}}{{{summary.worst_problem}}}",
        rf"\newcommand{{\AOSWorstDelta}}{{{summary.worst_delta:.3f}}}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_anytime_hv(anytime_csv: Path, *, out_pdf: Path, problems: list[str]) -> None:
    df = pd.read_csv(anytime_csv)
    if df.empty:
        raise ValueError(f"No rows in CSV: {anytime_csv}")

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"].isin(_VARIANT_ORDER)].copy()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)
    df["evals"] = df["evals"].astype(int)

    problems = [p.strip().lower() for p in problems if p.strip()]
    available = set(df["problem"].unique())
    missing = [p for p in problems if p not in available]
    if missing:
        raise ValueError(f"Problems not found in anytime CSV: {missing}")
    df = df[df["problem"].isin(problems)].copy()

    fig, axes = plt.subplots(1, len(problems), figsize=(4.8 * len(problems), 3.0), sharey=False)
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=True):
        grp = df[df["problem"] == problem]
        summary = (
            grp.groupby(["variant", "evals"], as_index=False)
            .agg(mean_hv=("hypervolume", "mean"), std_hv=("hypervolume", "std"))
            .sort_values(["variant", "evals"])
        )
        for variant in _VARIANT_ORDER:
            sub = summary[summary["variant"] == variant]
            if sub.empty:
                continue
            x = sub["evals"].to_numpy()
            y = sub["mean_hv"].to_numpy()
            yerr = sub["std_hv"].fillna(0.0).to_numpy()
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.6, capsize=2, label=_VARIANT_DISPLAY[variant])
        ax.set_title(problem.upper())
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Normalized HV")
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_operator_usage(trace_csv: Path, *, out_pdf: Path, checkpoints: list[int] | None = None) -> None:
    df = pd.read_csv(trace_csv)
    if df.empty:
        raise ValueError(f"No rows in CSV: {trace_csv}")

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"] == "aos"].copy()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)
    df["evals"] = df["evals"].astype(int)
    df["op_name"] = df["op_name"].astype(str).str.strip()

    checkpoints = checkpoints or [10000, 20000, 50000]
    checkpoints = sorted(set(int(c) for c in checkpoints if int(c) > 0))
    if not checkpoints:
        raise ValueError("checkpoints must contain at least one positive integer.")

    bounds = [0, *checkpoints]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        labels.append(f"<= {hi}" if lo <= 0 else f"{lo+1}-{hi}")
    labels.append(f"> {checkpoints[-1]}")

    def _assign_stage(evals: int) -> str:
        for hi, label in zip(bounds[1:], labels[:-1], strict=True):
            if evals <= hi:
                return label
        return labels[-1]

    df["stage"] = df["evals"].map(_assign_stage)

    pulls = (
        df.groupby(["problem", "seed", "stage", "op_name"], as_index=False)
        .size()
        .rename(columns={"size": "pulls"})
    )
    totals = pulls.groupby(["problem", "seed", "stage"], as_index=False)["pulls"].sum().rename(columns={"pulls": "total_pulls"})
    pulls = pulls.merge(totals, on=["problem", "seed", "stage"], how="left")
    pulls["fraction"] = pulls["pulls"] / pulls["total_pulls"].clip(lower=1)

    summary = (
        pulls.groupby(["problem", "stage", "op_name"], as_index=False)
        .agg(mean_fraction=("fraction", "mean"))
        .sort_values(["problem", "stage", "mean_fraction"], ascending=[True, True, False])
    )

    problems = sorted(summary["problem"].unique())
    stages = labels
    ops = sorted(summary["op_name"].unique())

    # Stable color mapping (max 5 in the committed portfolio)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(ops), 1)))
    color_by_op = {op: colors[i] for i, op in enumerate(ops)}

    fig, axes = plt.subplots(1, len(problems), figsize=(4.6 * len(problems), 3.1), sharey=True)
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=True):
        mat = (
            summary[summary["problem"] == problem]
            .pivot(index="stage", columns="op_name", values="mean_fraction")
            .reindex(index=stages, columns=ops)
            .fillna(0.0)
        )
        bottom = np.zeros(len(stages), dtype=float)
        x = np.arange(len(stages))
        for op in ops:
            y = mat[op].to_numpy()
            ax.bar(x, y, bottom=bottom, color=color_by_op[op], edgecolor="white", linewidth=0.4, label=op)
            bottom += y
        ax.set_title(problem.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean selection fraction")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=min(len(ops), 5), frameon=True, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "ablation_aos_racing_tuner.csv",
        help="AOS ablation CSV (baseline vs aos).",
    )
    ap.add_argument(
        "--anytime-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "ablation_aos_anytime.csv",
        help="Optional anytime HV CSV for convergence plots.",
    )
    ap.add_argument(
        "--trace-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "ablation_aos_trace.csv",
        help="Optional AOS trace CSV for operator-usage plots.",
    )
    ap.add_argument(
        "--n-evals",
        type=int,
        default=0,
        help="Filter to this evaluation budget (default: use max in CSV).",
    )
    ap.add_argument(
        "--problems",
        type=str,
        default="dtlz6,wfg3,zdt4",
        help="Comma-separated problems for convergence plots.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.ablation_csv.is_file():
        raise FileNotFoundError(f"Missing ablation CSV: {args.ablation_csv}")

    summary = summarize_ablation(args.ablation_csv, n_evals=int(args.n_evals))
    write_summary_macros(summary, TABLES_DIR / "summary_macros.tex")

    hv_caption = (
        r"Median normalized hypervolume by problem family (baseline NSGA-II vs.\ baseline + AOS). "
        r"Values computed from \texttt{experiments/ablation\_aos\_racing\_tuner.csv}."
    )
    runtime_caption = (
        r"Median runtime (seconds) by problem family (baseline NSGA-II vs.\ baseline + AOS). "
        r"Values computed from \texttt{experiments/ablation\_aos\_racing\_tuner.csv}."
    )

    hv_table = _make_family_table(
        summary.hv_family,
        caption=hv_caption,
        label="tab:hv_family",
        higher_is_better=True,
        decimals=3,
    )
    (TABLES_DIR / "ablation_hv_family.tex").write_text(hv_table + "\n", encoding="utf-8")

    runtime_table = _make_family_table(
        summary.runtime_family,
        caption=runtime_caption,
        label="tab:runtime_family",
        higher_is_better=False,
        decimals=2,
    )
    (TABLES_DIR / "ablation_runtime_family.tex").write_text(runtime_table + "\n", encoding="utf-8")

    hv_problem_caption = (
        r"Per-problem median normalized hypervolume at the final budget. "
        r"$\Delta$ denotes AOS minus baseline (medians over seeds)."
    )
    hv_problem_table = _make_per_problem_table(
        summary.hv_per_problem,
        caption=hv_problem_caption,
        label="tab:hv_per_problem",
        value_col="hypervolume",
        higher_is_better=True,
        decimals=3,
    )
    (TABLES_DIR / "ablation_hv_per_problem.tex").write_text(hv_problem_table + "\n", encoding="utf-8")

    rt_problem_caption = (
        r"Per-problem median runtime (seconds) at the final budget. "
        r"$\Delta$ denotes AOS minus baseline (medians over seeds)."
    )
    rt_problem_table = _make_per_problem_table(
        summary.runtime_per_problem.rename(columns={"runtime_seconds": "runtime_seconds"}),
        caption=rt_problem_caption,
        label="tab:runtime_per_problem",
        value_col="runtime_seconds",
        higher_is_better=False,
        decimals=2,
    )
    (TABLES_DIR / "ablation_runtime_per_problem.tex").write_text(rt_problem_table + "\n", encoding="utf-8")

    plot_delta_bars(
        summary.hv_per_problem,
        value_col="hypervolume",
        out_pdf=FIGURES_DIR / "hv_delta_by_problem.pdf",
        title="Median HV improvement by problem (AOS - baseline)",
        ylabel=r"$\Delta$ normalized HV",
    )

    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    if args.anytime_csv.is_file():
        plot_anytime_hv(args.anytime_csv, out_pdf=FIGURES_DIR / "anytime_hv_selected.pdf", problems=problems)

    if args.trace_csv.is_file():
        plot_operator_usage(args.trace_csv, out_pdf=FIGURES_DIR / "aos_operator_usage.pdf")


if __name__ == "__main__":
    main()

"""
Generate MIC paper assets (tables + figures) from the experiment CSVs.

Three-way comparison (baseline vs random vs AOS) on 21 standard MO
benchmarks (ZDT / DTLZ / WFG), grouped by benchmark family.

Usage:
  python paper_MIC/scripts/01_make_assets.py
  python paper_MIC/scripts/01_make_assets.py --ablation-csv experiments/mic/mic_ablation.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


ROOT_DIR = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT_DIR / "paper_MIC"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"


# ---------------------------------------------------------------------------
# Ordering constants
# ---------------------------------------------------------------------------

_FAMILY_ORDER = ("ZDT", "DTLZ", "WFG")
_VARIANT_ORDER = ("baseline", "random", "aos")
_VARIANT_DISPLAY = {
    "baseline": "Baseline",
    "random": "Random arm",
    "aos": "AOS",
}

_PROBLEM_ORDER = (
    # ZDT (2-obj, 10-30 vars)
    "zdt1", "zdt2", "zdt3", "zdt4", "zdt6",
    # DTLZ (3-obj, 7-22 vars)
    "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7",
    # WFG (3-obj, 24 vars)
    "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9",
)

_N_OBJ: dict[str, int] = {
    "zdt1": 2, "zdt2": 2, "zdt3": 2, "zdt4": 2, "zdt6": 2,
    "dtlz1": 3, "dtlz2": 3, "dtlz3": 3, "dtlz4": 3,
    "dtlz5": 3, "dtlz6": 3, "dtlz7": 3,
    "wfg1": 3, "wfg2": 3, "wfg3": 3, "wfg4": 3, "wfg5": 3,
    "wfg6": 3, "wfg7": 3, "wfg8": 3, "wfg9": 3,
}


def _family(problem_name: str) -> str:
    name = problem_name.strip().lower()
    if name.startswith("zdt"):
        return "ZDT"
    if name.startswith("dtlz"):
        return "DTLZ"
    if name.startswith("wfg"):
        return "WFG"
    return "Other"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_cell(value: float | None, decimals: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def _bold_best(cells: list[float | None], *, higher_is_better: bool, decimals: int) -> list[str]:
    """Format cells and bold the best value."""
    valid = [v for v in cells if v is not None]
    best = (max(valid) if higher_is_better else min(valid)) if valid else None
    result: list[str] = []
    for v in cells:
        text = _format_cell(v, decimals)
        if v is not None and best is not None and abs(v - best) < 1e-12:
            text = rf"\textbf{{{text}}}"
        result.append(text)
    return result


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def _make_family_table(
    group_df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    higher_is_better: bool,
    decimals: int,
) -> str:
    """Family-level table grouped by benchmark suite (ZDT / DTLZ / WFG)."""
    families = [f for f in _FAMILY_ORDER if f in group_df.columns]
    cols = [*families, "Average"]

    best_by_col: dict[str, float | None] = {}
    for col in cols:
        if col not in group_df.columns:
            best_by_col[col] = None
            continue
        series = group_df[col].dropna()
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
        if variant not in group_df.index:
            continue
        row = group_df.loc[variant]
        display = _VARIANT_DISPLAY.get(variant, variant)

        cells: list[str] = []
        for col in cols:
            v: float | None = None
            if col in group_df.columns:
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
    """Per-problem table with all three variants and deltas."""
    pivot = df.pivot(index="problem", columns="variant", values=value_col)
    pivot = pivot.reindex(index=list(_PROBLEM_ORDER)).dropna(how="all")
    # Deltas: AOS vs baseline, AOS vs random
    if "aos" in pivot.columns and "baseline" in pivot.columns:
        pivot["d_base"] = pivot["aos"] - pivot["baseline"]
    if "aos" in pivot.columns and "random" in pivot.columns:
        pivot["d_rand"] = pivot["aos"] - pivot["random"]

    header_cols = [v for v in _VARIANT_ORDER if v in pivot.columns]
    n_data = len(header_cols)
    delta_cols = [c for c in ["d_base", "d_rand"] if c in pivot.columns]
    delta_labels = {
        "d_base": r"$\Delta_{\text{base}}$",
        "d_rand": r"$\Delta_{\text{rand}}$",
    }

    col_spec = "r" * n_data + ("r" * len(delta_cols))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{l|{col_spec}}}",
        r"\toprule",
        r"\textbf{Problem} & "
        + " & ".join(rf"\textbf{{{_VARIANT_DISPLAY.get(v, v)}}}" for v in header_cols)
        + (" & " + " & ".join(delta_labels[c] for c in delta_cols) if delta_cols else "")
        + r" \\",
        r"\midrule",
    ]

    last_family: str | None = None
    for problem in pivot.index:
        fam = _family(problem)
        if last_family is not None and fam != last_family:
            lines.append(r"\midrule")
        last_family = fam

        values = [
            None if pd.isna(pivot.loc[problem].get(v)) else float(pivot.loc[problem][v])
            for v in header_cols
        ]
        bolded = _bold_best(values, higher_is_better=higher_is_better, decimals=decimals)

        delta_cells = []
        for c in delta_cols:
            raw = pivot.loc[problem].get(c)
            delta_cells.append(_format_cell(None if pd.isna(raw) else float(raw), decimals))

        lines.append(f"{problem} & " + " & ".join(bolded) + (" & " + " & ".join(delta_cells) if delta_cells else "") + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_delta_bars(
    per_problem: pd.DataFrame,
    *,
    value_col: str,
    out_pdf: Path,
    title: str,
    ylabel: str,
    reference_variant: str = "baseline",
) -> None:
    """Bar chart of HV delta (AOS - reference) per problem."""
    df = per_problem.pivot(index="problem", columns="variant", values=value_col).copy()
    df = df.reindex(index=list(_PROBLEM_ORDER)).dropna(how="all")
    if "aos" not in df.columns or reference_variant not in df.columns:
        return
    df["delta"] = df["aos"] - df[reference_variant]
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


def plot_anytime_hv(anytime_csv: Path, *, out_pdf: Path, problems: list[str]) -> None:
    """Convergence curves (mean +/- std) for selected problems."""
    df = pd.read_csv(anytime_csv)
    if df.empty:
        raise ValueError(f"Empty CSV: {anytime_csv}")

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"].isin(_VARIANT_ORDER)].copy()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)
    df["evals"] = df["evals"].astype(int)

    problems = [p.strip().lower() for p in problems if p.strip()]
    available = set(df["problem"].unique())
    missing = [p for p in problems if p not in available]
    if missing:
        print(f"  WARNING: problems missing from anytime CSV: {missing}")
        problems = [p for p in problems if p in available]
    if not problems:
        return
    df = df[df["problem"].isin(problems)].copy()

    style_map = {
        "baseline": {"color": "#1f77b4", "linestyle": "-"},
        "random": {"color": "#ff7f0e", "linestyle": "--"},
        "aos": {"color": "#2ca02c", "linestyle": "-"},
    }

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
            st = style_map.get(variant, {})
            ax.errorbar(
                x, y, yerr=yerr,
                marker="o", linewidth=1.6, capsize=2,
                label=_VARIANT_DISPLAY.get(variant, variant),
                **st,
            )
        ax.set_title(problem.upper())
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Normalized HV")
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_anytime_aggregate(anytime_csv: Path, *, out_pdf: Path) -> None:
    """Aggregate convergence curve averaged across ALL problems."""
    df = pd.read_csv(anytime_csv)
    if df.empty:
        return

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"].isin(_VARIANT_ORDER)].copy()
    df["evals"] = df["evals"].astype(int)

    agg = df.groupby(["variant", "evals"], as_index=False).agg(
        mean_hv=("hypervolume", "mean"),
        std_hv=("hypervolume", "std"),
    )

    style_map = {
        "baseline": {"color": "#1f77b4", "linestyle": "-"},
        "random": {"color": "#ff7f0e", "linestyle": "--"},
        "aos": {"color": "#2ca02c", "linestyle": "-"},
    }

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
    for variant in _VARIANT_ORDER:
        sub = agg[agg["variant"] == variant].sort_values("evals")
        if sub.empty:
            continue
        x = sub["evals"].to_numpy()
        y = sub["mean_hv"].to_numpy()
        yerr = sub["std_hv"].fillna(0.0).to_numpy()
        st = style_map.get(variant, {})
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", linewidth=2.0, capsize=3,
            label=_VARIANT_DISPLAY.get(variant, variant),
            **st,
        )
    ax.set_title("Mean normalized HV across all 21 problems")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Mean normalized HV")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_operator_usage(trace_csv: Path, *, out_pdf: Path, checkpoints: list[int] | None = None) -> None:
    """Stacked bar chart of operator selection fractions by search stage."""
    df = pd.read_csv(trace_csv)
    if df.empty:
        raise ValueError(f"Empty CSV: {trace_csv}")

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
        labels.append(f"<= {hi}" if lo <= 0 else f"{lo + 1}-{hi}")
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


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatTestResult:
    """Win/Tie/Loss counts from Wilcoxon + Holm-Bonferroni across problems."""
    wins: int
    ties: int
    losses: int
    per_problem: dict[str, dict]  # problem -> {p_value, corrected_reject, direction, a_measure}


def _vargha_delaney(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A-measure (effect size).  A > 0.5 means x > y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.5
    r = sp_stats.rankdata(np.concatenate([x, y]))
    r1 = r[:nx].sum()
    return (r1 / nx - (nx + 1) / 2) / ny


def _holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list of reject booleans using Holm-Bonferroni step-down correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    reject = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            break
    return reject


def run_stat_tests(
    df: pd.DataFrame,
    *,
    variant_a: str = "aos",
    variant_b: str = "baseline",
    value_col: str = "hypervolume",
    alpha: float = 0.05,
) -> StatTestResult:
    """Wilcoxon signed-rank test per problem, with Holm-Bonferroni correction."""
    problems = sorted(df["problem"].unique())
    p_values: list[float] = []
    directions: list[int] = []  # +1 = a > b, -1 = a < b, 0 = tied
    a_measures: list[float] = []
    problem_order: list[str] = []

    for problem in problems:
        a_vals = df[(df["problem"] == problem) & (df["variant"] == variant_a)][value_col].to_numpy()
        b_vals = df[(df["problem"] == problem) & (df["variant"] == variant_b)][value_col].to_numpy()
        if len(a_vals) == 0 or len(b_vals) == 0:
            continue
        n = min(len(a_vals), len(b_vals))
        a_vals, b_vals = a_vals[:n], b_vals[:n]
        diff = a_vals - b_vals
        if np.allclose(diff, 0.0):
            p_values.append(1.0)
            directions.append(0)
        else:
            try:
                stat, p = sp_stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
                p_values.append(float(p))
                directions.append(1 if np.median(a_vals) > np.median(b_vals) else -1)
            except ValueError:
                p_values.append(1.0)
                directions.append(0)
        a_measures.append(_vargha_delaney(a_vals, b_vals))
        problem_order.append(problem)

    rejects = _holm_bonferroni(p_values, alpha=alpha)

    wins, ties, losses = 0, 0, 0
    per_problem: dict[str, dict] = {}
    for i, problem in enumerate(problem_order):
        if rejects[i]:
            if directions[i] > 0:
                wins += 1
                label = "win"
            else:
                losses += 1
                label = "loss"
        else:
            ties += 1
            label = "tie"
        per_problem[problem] = {
            "p_value": p_values[i],
            "corrected_reject": rejects[i],
            "direction": label,
            "a_measure": a_measures[i],
        }

    return StatTestResult(wins=wins, ties=ties, losses=losses, per_problem=per_problem)


@dataclass(frozen=True)
class AblationSummary:
    hv_family: pd.DataFrame
    runtime_family: pd.DataFrame
    hv_per_problem: pd.DataFrame
    runtime_per_problem: pd.DataFrame
    hv_overall_mean: dict[str, float]
    runtime_overall_median: dict[str, float]
    n_problems: int
    n_seeds: int
    # AOS vs baseline
    wins_vs_base: int
    losses_vs_base: int
    best_problem_vs_base: str
    best_delta_vs_base: float
    worst_problem_vs_base: str
    worst_delta_vs_base: float
    # AOS vs random
    wins_vs_rand: int
    losses_vs_rand: int
    # Convergence speed
    hv_mean_at_10k: dict[str, float]
    convergence_advantage_pct: float  # AOS vs baseline at 10k evals
    # Statistical tests
    stat_vs_base: StatTestResult | None = field(default=None)
    stat_vs_rand: StatTestResult | None = field(default=None)


def summarize_ablation(csv_path: Path, *, n_evals: int = 0, anytime_csv: Path | None = None) -> AblationSummary:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows in CSV: {csv_path}")

    if "n_evals" in df.columns:
        df["n_evals"] = df["n_evals"].astype(int)
        available = sorted(int(x) for x in df["n_evals"].dropna().unique())
        if not available:
            raise ValueError("CSV has n_evals column but no values.")
        target = int(n_evals) if int(n_evals) > 0 else int(max(available))
        if target not in available:
            raise ValueError(f"n_evals={target} not present (available: {available}).")
        df = df[df["n_evals"] == target].copy()

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"].isin(_VARIANT_ORDER)].copy()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)

    # Assign family group
    df["family"] = df["problem"].map(_family)

    # Family-level means
    hv_family = df.groupby(["variant", "family"])["hypervolume"].mean().unstack()
    hv_family["Average"] = hv_family.mean(axis=1)
    hv_family = hv_family.reindex(index=list(_VARIANT_ORDER))

    runtime_family = df.groupby(["variant", "family"])["runtime_seconds"].median().unstack()
    runtime_family["Average"] = runtime_family.mean(axis=1)
    runtime_family = runtime_family.reindex(index=list(_VARIANT_ORDER))

    # Overall
    hv_overall = df.groupby("variant")["hypervolume"].mean().to_dict()
    rt_overall = df.groupby("variant")["runtime_seconds"].median().to_dict()

    # Per-problem means
    hv_pp = df.groupby(["variant", "problem"], as_index=False)["hypervolume"].mean()
    rt_pp = df.groupby(["variant", "problem"], as_index=False)["runtime_seconds"].median()

    # AOS vs baseline
    hv_wide = hv_pp.pivot(index="problem", columns="variant", values="hypervolume")
    delta_base = hv_wide.get("aos") - hv_wide.get("baseline") if {"aos", "baseline"} <= set(hv_wide.columns) else pd.Series(dtype=float)
    wins_b = int((delta_base > 0).sum()) if delta_base.size else 0
    losses_b = int((delta_base < 0).sum()) if delta_base.size else 0
    best_pb = str(delta_base.idxmax()) if delta_base.size else ""
    best_db = float(delta_base.max()) if delta_base.size else 0.0
    worst_pb = str(delta_base.idxmin()) if delta_base.size else ""
    worst_db = float(delta_base.min()) if delta_base.size else 0.0

    # AOS vs random
    delta_rand = hv_wide.get("aos") - hv_wide.get("random") if {"aos", "random"} <= set(hv_wide.columns) else pd.Series(dtype=float)
    wins_r = int((delta_rand > 0).sum()) if delta_rand.size else 0
    losses_r = int((delta_rand < 0).sum()) if delta_rand.size else 0

    # Convergence speed from anytime data
    hv_at_10k: dict[str, float] = {}
    conv_adv = 0.0
    if anytime_csv is not None and anytime_csv.is_file():
        at = pd.read_csv(anytime_csv)
        at["variant"] = at["variant"].astype(str).str.strip().str.lower()
        at["evals"] = at["evals"].astype(int)
        at_10k = at[at["evals"] == 10000]
        if not at_10k.empty:
            hv_at_10k = at_10k.groupby("variant")["hypervolume"].mean().to_dict()
            bl_10k = hv_at_10k.get("baseline", 0.0)
            ao_10k = hv_at_10k.get("aos", 0.0)
            conv_adv = ((ao_10k - bl_10k) / max(bl_10k, 1e-12)) * 100.0

    # Statistical tests
    stat_base = run_stat_tests(df, variant_a="aos", variant_b="baseline") if "aos" in df["variant"].values and "baseline" in df["variant"].values else None
    stat_rand = run_stat_tests(df, variant_a="aos", variant_b="random") if "aos" in df["variant"].values and "random" in df["variant"].values else None

    return AblationSummary(
        hv_family=hv_family,
        runtime_family=runtime_family,
        hv_per_problem=hv_pp,
        runtime_per_problem=rt_pp,
        hv_overall_mean={str(k): float(v) for k, v in hv_overall.items()},
        runtime_overall_median={str(k): float(v) for k, v in rt_overall.items()},
        n_problems=int(hv_wide.shape[0]),
        n_seeds=int(df["seed"].nunique()),
        wins_vs_base=wins_b,
        losses_vs_base=losses_b,
        best_problem_vs_base=best_pb,
        best_delta_vs_base=best_db,
        worst_problem_vs_base=worst_pb,
        worst_delta_vs_base=worst_db,
        wins_vs_rand=wins_r,
        losses_vs_rand=losses_r,
        hv_mean_at_10k=hv_at_10k,
        convergence_advantage_pct=conv_adv,
        stat_vs_base=stat_base,
        stat_vs_rand=stat_rand,
    )


def write_summary_macros(summary: AblationSummary, out_path: Path) -> None:
    hv_base = float(summary.hv_overall_mean.get("baseline", 0.0))
    hv_rand = float(summary.hv_overall_mean.get("random", 0.0))
    hv_aos = float(summary.hv_overall_mean.get("aos", 0.0))
    rt_base = float(summary.runtime_overall_median.get("baseline", 1.0))
    rt_rand = float(summary.runtime_overall_median.get("random", 1.0))
    rt_aos = float(summary.runtime_overall_median.get("aos", 1.0))
    rt_overhead = 100.0 * (rt_aos - rt_base) / max(rt_base, 1e-12)

    # Convergence
    hv_base_10k = float(summary.hv_mean_at_10k.get("baseline", 0.0))
    hv_aos_10k = float(summary.hv_mean_at_10k.get("aos", 0.0))

    # Statistical test results
    sb = summary.stat_vs_base
    sr = summary.stat_vs_rand
    stat_wins_b = sb.wins if sb else 0
    stat_ties_b = sb.ties if sb else summary.n_problems
    stat_losses_b = sb.losses if sb else 0
    stat_wins_r = sr.wins if sr else 0
    stat_ties_r = sr.ties if sr else summary.n_problems
    stat_losses_r = sr.losses if sr else 0

    lines = [
        "% Auto-generated by paper_MIC/scripts/01_make_assets.py",
        rf"\newcommand{{\AOSNProblems}}{{{summary.n_problems}}}",
        rf"\newcommand{{\AOSNSeeds}}{{{summary.n_seeds}}}",
        rf"\newcommand{{\AOSWins}}{{{summary.wins_vs_base}}}",
        rf"\newcommand{{\AOSLosses}}{{{summary.losses_vs_base}}}",
        rf"\newcommand{{\AOSWinsRand}}{{{summary.wins_vs_rand}}}",
        rf"\newcommand{{\AOSLossesRand}}{{{summary.losses_vs_rand}}}",
        rf"\newcommand{{\AOSHVMeanBaseline}}{{{hv_base:.3f}}}",
        rf"\newcommand{{\AOSHVMeanRandom}}{{{hv_rand:.3f}}}",
        rf"\newcommand{{\AOSHVMeanAOS}}{{{hv_aos:.3f}}}",
        rf"\newcommand{{\AOSRuntimeMedianBaseline}}{{{rt_base:.2f}}}",
        rf"\newcommand{{\AOSRuntimeMedianRandom}}{{{rt_rand:.2f}}}",
        rf"\newcommand{{\AOSRuntimeMedianAOS}}{{{rt_aos:.2f}}}",
        rf"\newcommand{{\AOSRuntimeOverheadPct}}{{{rt_overhead:.0f}}}",
        rf"\newcommand{{\AOSBestProblem}}{{{summary.best_problem_vs_base}}}",
        rf"\newcommand{{\AOSBestDelta}}{{{summary.best_delta_vs_base:.3f}}}",
        rf"\newcommand{{\AOSWorstProblem}}{{{summary.worst_problem_vs_base}}}",
        rf"\newcommand{{\AOSWorstDelta}}{{{summary.worst_delta_vs_base:.3f}}}",
        "% Convergence speed",
        rf"\newcommand{{\AOSHVBaselineTenK}}{{{hv_base_10k:.3f}}}",
        rf"\newcommand{{\AOSHVAOSTenK}}{{{hv_aos_10k:.3f}}}",
        rf"\newcommand{{\AOSConvergenceAdvPct}}{{{summary.convergence_advantage_pct:.0f}}}",
        "% Statistical test results (Wilcoxon + Holm-Bonferroni)",
        rf"\newcommand{{\StatWinsBase}}{{{stat_wins_b}}}",
        rf"\newcommand{{\StatTiesBase}}{{{stat_ties_b}}}",
        rf"\newcommand{{\StatLossesBase}}{{{stat_losses_b}}}",
        rf"\newcommand{{\StatWinsRand}}{{{stat_wins_r}}}",
        rf"\newcommand{{\StatTiesRand}}{{{stat_ties_r}}}",
        rf"\newcommand{{\StatLossesRand}}{{{stat_losses_r}}}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_stat_table(summary: AblationSummary, out_path: Path) -> None:
    """Write the statistical significance table."""
    sb = summary.stat_vs_base
    sr = summary.stat_vs_rand
    w_b = sb.wins if sb else 0
    t_b = sb.ties if sb else summary.n_problems
    l_b = sb.losses if sb else 0
    w_r = sr.wins if sr else 0
    t_r = sr.ties if sr else summary.n_problems
    l_r = sr.losses if sr else 0

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Win/Tie/Loss counts (Wilcoxon signed-rank, $\alpha{=}0.05$, Holm--Bonferroni corrected). "
        r"A ``win'' means AOS is significantly better; ``loss'' means significantly worse.}",
        r"\label{tab:stat_test}",
        r"\begin{tabular}{l|ccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Win} & \textbf{Tie} & \textbf{Loss} \\",
        r"\midrule",
        rf"AOS vs.\ Baseline   & {w_b} & {t_b} & {l_b} \\",
        rf"AOS vs.\ Random arm & {w_r} & {t_r} & {l_r} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_reward_evolution(trace_csv: Path, *, out_pdf: Path, problem: str = "dtlz1", seed: int = 0) -> None:
    """Plot per-generation reward and selected arm for a representative run."""
    df = pd.read_csv(trace_csv)
    if df.empty:
        return

    df["variant"] = df["variant"].astype(str).str.strip().str.lower()
    df = df[df["variant"] == "aos"].copy()
    df["problem"] = df["problem"].astype(str).str.strip().str.lower()
    df["seed"] = df["seed"].astype(int)

    sub = df[(df["problem"] == problem.lower()) & (df["seed"] == seed)].copy()
    if sub.empty:
        avail = df[df["problem"] == problem.lower()]["seed"].unique()
        if len(avail) == 0:
            print(f"  WARNING: no trace data for {problem}, skipping reward evolution")
            return
        sub = df[(df["problem"] == problem.lower()) & (df["seed"] == int(avail[0]))].copy()

    sub = sub.sort_values("evals")
    if "reward" not in sub.columns or "op_name" not in sub.columns:
        print(f"  WARNING: trace CSV lacks 'reward' or 'op_name' column, skipping reward evolution")
        return

    evals = sub["evals"].to_numpy()
    reward = sub["reward"].to_numpy(dtype=float) if "reward" in sub.columns else np.zeros(len(sub))
    op_names = sub["op_name"].astype(str).to_numpy()

    unique_ops = sorted(set(op_names))
    op_to_idx = {op: i for i, op in enumerate(unique_ops)}
    arm_idx = np.array([op_to_idx[op] for op in op_names])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(evals, reward, color="#1f77b4", linewidth=0.7, alpha=0.8)
    if len(reward) > 20:
        window = max(5, len(reward) // 20)
        smoothed = pd.Series(reward).rolling(window, min_periods=1, center=True).mean().to_numpy()
        ax1.plot(evals, smoothed, color="#d62728", linewidth=2.0, label=f"rolling avg (w={window})")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Reward")
    ax1.set_title(f"Reward evolution and arm selection ({problem.upper()}, seed {seed})")
    ax1.grid(True, alpha=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_ops), 1)))
    for op, idx in op_to_idx.items():
        mask = arm_idx == idx
        ax2.scatter(evals[mask], arm_idx[mask], color=colors[idx], s=4, alpha=0.6, label=op)
    ax2.set_yticks(list(range(len(unique_ops))))
    ax2.set_yticklabels(unique_ops, fontsize=7)
    ax2.set_ylabel("Selected arm")
    ax2.set_xlabel("Evaluations")
    ax2.legend(loc="upper right", fontsize=7, ncol=2, markerscale=3)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "mic" / "mic_ablation.csv",
        help="MIC ablation CSV (baseline vs random vs AOS).",
    )
    ap.add_argument(
        "--anytime-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "mic" / "mic_anytime.csv",
        help="Optional anytime HV CSV for convergence plots.",
    )
    ap.add_argument(
        "--trace-csv",
        type=Path,
        default=ROOT_DIR / "experiments" / "mic" / "mic_trace.csv",
        help="Optional AOS trace CSV for operator-usage plots.",
    )
    ap.add_argument(
        "--n-evals",
        type=int,
        default=0,
        help="Filter to this evaluation budget (default: max in CSV).",
    )
    ap.add_argument(
        "--problems",
        type=str,
        default="zdt2,dtlz7,wfg1,wfg2",
        help="Comma-separated problems for convergence plots.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.ablation_csv.is_file():
        raise FileNotFoundError(f"Missing ablation CSV: {args.ablation_csv}")

    anytime_path = args.anytime_csv if args.anytime_csv.is_file() else None
    summary = summarize_ablation(args.ablation_csv, n_evals=int(args.n_evals), anytime_csv=anytime_path)
    write_summary_macros(summary, TABLES_DIR / "summary_macros.tex")

    # --- Family-level tables ---
    hv_caption = (
        r"Mean normalized hypervolume by benchmark family "
        r"(baseline vs.\ random arm vs.\ AOS, Thompson Sampling)."
    )
    runtime_caption = (
        r"Median runtime (seconds) by benchmark family "
        r"(baseline vs.\ random arm vs.\ AOS)."
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

    # --- Per-problem tables ---
    hv_pp_caption = (
        r"Per-problem mean normalized hypervolume. "
        r"$\Delta_{\text{base}}$: AOS $-$ baseline; "
        r"$\Delta_{\text{rand}}$: AOS $-$ random arm."
    )
    hv_pp = _make_per_problem_table(
        summary.hv_per_problem,
        caption=hv_pp_caption,
        label="tab:hv_per_problem",
        value_col="hypervolume",
        higher_is_better=True,
        decimals=3,
    )
    (TABLES_DIR / "ablation_hv_per_problem.tex").write_text(hv_pp + "\n", encoding="utf-8")

    rt_pp_caption = (
        r"Per-problem median runtime (seconds). "
        r"$\Delta_{\text{base}}$: AOS $-$ baseline; "
        r"$\Delta_{\text{rand}}$: AOS $-$ random arm."
    )
    rt_pp = _make_per_problem_table(
        summary.runtime_per_problem,
        caption=rt_pp_caption,
        label="tab:runtime_per_problem",
        value_col="runtime_seconds",
        higher_is_better=False,
        decimals=2,
    )
    (TABLES_DIR / "ablation_runtime_per_problem.tex").write_text(rt_pp + "\n", encoding="utf-8")

    # --- Figures ---
    plot_delta_bars(
        summary.hv_per_problem,
        value_col="hypervolume",
        out_pdf=FIGURES_DIR / "hv_delta_by_problem.pdf",
        title="Mean HV improvement by problem (AOS $-$ baseline)",
        ylabel=r"$\Delta$ normalized HV",
        reference_variant="baseline",
    )
    plot_delta_bars(
        summary.hv_per_problem,
        value_col="hypervolume",
        out_pdf=FIGURES_DIR / "hv_delta_vs_random.pdf",
        title="Mean HV improvement by problem (AOS $-$ random arm)",
        ylabel=r"$\Delta$ normalized HV",
        reference_variant="random",
    )

    # --- Statistical significance table ---
    write_stat_table(summary, TABLES_DIR / "stat_test.tex")

    # Convergence plots
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    if anytime_path is not None:
        plot_anytime_hv(anytime_path, out_pdf=FIGURES_DIR / "anytime_hv_selected.pdf", problems=problems)
        plot_anytime_aggregate(anytime_path, out_pdf=FIGURES_DIR / "anytime_hv_aggregate.pdf")

    # Operator usage
    if args.trace_csv.is_file():
        plot_operator_usage(args.trace_csv, out_pdf=FIGURES_DIR / "aos_operator_usage.pdf")
        plot_reward_evolution(args.trace_csv, out_pdf=FIGURES_DIR / "reward_evolution.pdf", problem="dtlz1")

    print(f"Assets written to {TABLES_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()

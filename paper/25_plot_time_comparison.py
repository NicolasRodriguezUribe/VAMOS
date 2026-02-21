"""
Generate the runtime comparison bar chart (time_comparison.png).

Reads experiments/benchmark_paper.csv and plots the ratio of each competitor's
median family-level runtime to VAMOS (Numba), grouped by problem family.

Output: paper/manuscript/figures/time_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
CSV_PATH = ROOT_DIR / "experiments" / "benchmark_paper.csv"
OUTPUT_FIG = Path(__file__).parent / "manuscript" / "figures" / "time_comparison.png"

# VAMOS backend used as baseline
VAMOS_BACKEND = "VAMOS (numba)"

# Competitors to show (order = left to right in chart)
COMPETITORS = ["jMetalPy", "pymoo"]

FAMILIES = ["ZDT", "DTLZ", "WFG"]
FAMILY_COLORS = {"ZDT": "#1f77b4", "DTLZ": "#e8951d", "WFG": "#2ca02c"}


def get_family(problem: str) -> str:
    if problem.startswith("zdt"):
        return "ZDT"
    if problem.startswith("dtlz"):
        return "DTLZ"
    if problem.startswith("wfg"):
        return "WFG"
    return "Other"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["family"] = df["problem"].apply(get_family)

    # Compute per-problem median runtime across seeds
    per_problem = (
        df.groupby(["framework", "problem", "family"])["runtime_seconds"]
        .median()
        .reset_index()
        .rename(columns={"runtime_seconds": "median_runtime"})
    )

    # VAMOS baseline: prefer numba, fall back to moocore then numpy per problem
    vamos_backends = [VAMOS_BACKEND, "VAMOS (moocore)", "VAMOS (numpy)"]
    vamos_rows = []
    for prob in per_problem["problem"].unique():
        for vb in vamos_backends:
            row = per_problem[(per_problem["framework"] == vb) & (per_problem["problem"] == prob)]
            if not row.empty:
                vamos_rows.append(row.iloc[0])
                break
    vamos = pd.DataFrame(vamos_rows)
    vamos = vamos.rename(columns={"median_runtime": "vamos_runtime"})
    vamos = vamos[["problem", "family", "vamos_runtime"]]

    # Build ratio data for each competitor × family
    bar_data = []  # list of (competitor, family, median_ratio, lo_err, hi_err)
    for comp in COMPETITORS:
        comp_data = per_problem[per_problem["framework"] == comp].copy()
        merged = comp_data.merge(vamos, on=["problem", "family"])
        merged["ratio"] = merged["median_runtime"] / merged["vamos_runtime"]

        for fam in FAMILIES:
            fam_ratios = merged[merged["family"] == fam]["ratio"]
            if fam_ratios.empty:
                bar_data.append((comp, fam, np.nan, 0, 0))
                continue
            med = fam_ratios.median()
            q25 = fam_ratios.quantile(0.25)
            q75 = fam_ratios.quantile(0.75)
            bar_data.append((comp, fam, med, med - q25, q75 - med))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    n_comps = len(COMPETITORS)
    n_fams = len(FAMILIES)
    bar_width = 0.22
    group_width = n_fams * bar_width + 0.1

    for i, comp in enumerate(COMPETITORS):
        group_center = i * group_width
        for j, fam in enumerate(FAMILIES):
            entry = [e for e in bar_data if e[0] == comp and e[1] == fam][0]
            _, _, med, lo, hi = entry
            x = group_center + (j - (n_fams - 1) / 2) * bar_width
            ax.bar(
                x,
                med,
                width=bar_width * 0.85,
                color=FAMILY_COLORS[fam],
                label=fam if i == 0 else None,
                yerr=[[lo], [hi]],
                capsize=3,
                edgecolor="none",
                error_kw={"linewidth": 1},
            )

    # Parity line
    xlim = (-0.5, (n_comps - 1) * group_width + 0.5)
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlim(*xlim)

    # Log scale
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Runtime relative to VAMOS (numba)")

    # Y ticks
    yticks = [1, 2, 4, 8, 16, 32]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$2^{{{int(np.log2(y))}}}$" for y in yticks])
    ax.yaxis.grid(True, which="major", linestyle=":", alpha=0.5)

    # X ticks
    ax.set_xticks([i * group_width for i in range(n_comps)])
    ax.set_xticklabels(COMPETITORS, fontsize=12)

    ax.legend(title="Problem Family", loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_FIG}")
    plt.close(fig)


if __name__ == "__main__":
    main()

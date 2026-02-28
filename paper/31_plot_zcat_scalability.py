"""
ZCAT Scalability Bar Chart for VAMOS Paper
==========================================
Creates a grouped bar chart showing median runtime of NSGA-II on ZCAT1–20
across frameworks (VAMOS, pymoo, jMetalPy) with varying objective counts.

Usage: python paper/31_plot_zcat_scalability.py

Reads:  experiments/benchmark_zcat_scalability.csv
Writes: paper/manuscript/figures/zcat_scalability.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "experiments" / "benchmark_zcat_scalability.csv"
OUTPUT_FIG = Path(__file__).parent / "manuscript" / "figures" / "zcat_scalability.png"

FRAMEWORK_STYLES = {
    "VAMOS (numba)": {"color": "#1f77b4", "label": "VAMOS"},
    "pymoo": {"color": "#ff7f0e", "label": "pymoo"},
    "jMetalPy": {"color": "#2ca02c", "label": "jMetalPy"},
}


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run 29_run_zcat_scalability.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Frameworks: {df['framework'].unique().tolist()}")
    print(f"Objectives: {sorted(df['n_obj'].unique())}")
    print(f"Problems: {len(df['problem'].unique())}")

    obj_counts = sorted(df["n_obj"].unique())
    frameworks = list(FRAMEWORK_STYLES.keys())
    n_groups = len(obj_counts)
    n_bars = len(frameworks)
    bar_width = 0.22
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, fw in enumerate(frameworks):
        style = FRAMEWORK_STYLES[fw]
        medians = []
        q25s = []
        q75s = []

        for n_obj in obj_counts:
            subset = df[(df["framework"] == fw) & (df["n_obj"] == n_obj)]
            # Per-problem median runtime, then aggregate across problems
            per_problem = subset.groupby("problem")["runtime_seconds"].median()
            medians.append(per_problem.median())
            q25s.append(per_problem.quantile(0.25))
            q75s.append(per_problem.quantile(0.75))

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        yerr_lo = medians - q25s
        yerr_hi = q75s - medians

        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, medians, bar_width,
            label=style["label"], color=style["color"], alpha=0.85,
            yerr=[yerr_lo, yerr_hi], capsize=3, error_kw={"linewidth": 0.8},
        )

        # Annotate bar values
        for bar, val in zip(bars, medians):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Number of objectives", fontsize=10)
    ax.set_ylabel("Median runtime (seconds)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in obj_counts])
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_FIG}")


if __name__ == "__main__":
    main()

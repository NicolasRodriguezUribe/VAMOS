"""
Scaling Plot Generator for VAMOS Paper
======================================
Creates a multi-panel figure showing runtime scaling of VAMOS backends
(Numba vs NumPy) as population size increases.

Usage: python paper/30_plot_scaling.py

Reads:  experiments/scaling_vectorization.csv
Writes: paper/generated/figures/scaling.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "experiments" / "scaling_vectorization.csv"
OUTPUT_FIG = Path(__file__).parent / "generated" / "figures" / "scaling.png"

ENGINE_STYLES = {
    "numba": {"color": "#1f77b4", "label": "Numba", "ls": "-", "marker": "o"},
    "numpy": {"color": "#ff7f0e", "label": "NumPy", "ls": "--", "marker": "s"},
}

PROBLEM_TITLES = {
    "zdt4": "ZDT4",
    "dtlz2": "DTLZ2",
    "wfg2": "WFG2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the manuscript scaling figure.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FIG)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated figure.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}. Pass --overwrite to replace it.")
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run 03_run_scaling_experiment.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    pop_df = df[df["experiment"] == "population"].copy()
    problems = sorted(pop_df["problem"].unique())
    n_panels = len(problems)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5), squeeze=False)
    axes = axes.flatten()

    for i, problem in enumerate(problems):
        ax = axes[i]
        problem_data = pop_df[pop_df["problem"] == problem]

        for engine, style in ENGINE_STYLES.items():
            eng_data = problem_data[problem_data["engine"] == engine]
            if eng_data.empty:
                continue

            grouped = eng_data.groupby("pop_size")["runtime_seconds"]
            medians = grouped.median()
            q25 = grouped.quantile(0.25)
            q75 = grouped.quantile(0.75)

            pop_sizes = np.array(medians.index, dtype=float)
            ax.plot(pop_sizes, medians.values, color=style["color"], ls=style["ls"],
                    marker=style["marker"], label=style["label"], linewidth=1.5, markersize=5)
            ax.fill_between(pop_sizes, q25.values, q75.values, color=style["color"], alpha=0.15)

        title = PROBLEM_TITLES.get(problem, problem.upper())
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Population size", fontsize=9)
        if i == 0:
            ax.set_ylabel("Runtime (seconds)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ENGINE_STYLES),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scaling figure to {args.output}")


if __name__ == "__main__":
    main()

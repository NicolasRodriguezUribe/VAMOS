"""
Generate memory footprint bar chart comparing VAMOS vs pymoo vs jMetalPy.

Usage: python paper/26_plot_memory_comparison.py

Reads:  experiments/memory_benchmark.csv
Writes: paper/manuscript/figures/memory_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "experiments"
FIG_DIR = Path(__file__).parent / "manuscript" / "figures"

INPUT_CSV = DATA_DIR / "memory_benchmark.csv"
OUTPUT_PNG = FIG_DIR / "memory_comparison.png"

# Display order and colours (match time_comparison.py style)
FRAMEWORK_ORDER = ["VAMOS", "pymoo", "jMetalPy"]
FRAMEWORK_COLORS = {"VAMOS": "#4C72B0", "pymoo": "#DD8452", "jMetalPy": "#55A868"}
PROBLEM_ORDER = ["zdt1", "dtlz2", "wfg4"]
PROBLEM_LABELS = {"zdt1": "ZDT1", "dtlz2": "DTLZ2", "wfg4": "WFG4"}


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # Normalise VAMOS backend name
    df["framework"] = df["framework"].replace({"VAMOS (numba)": "VAMOS"})

    # Compute median peak memory per (framework, problem)
    med = df.groupby(["framework", "problem"])["peak_memory_mb"].median().reset_index()

    # Build grouped bar chart
    n_problems = len(PROBLEM_ORDER)
    n_frameworks = len(FRAMEWORK_ORDER)
    x = np.arange(n_problems)
    width = 0.22

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for i, fw in enumerate(FRAMEWORK_ORDER):
        vals = []
        for prob in PROBLEM_ORDER:
            row = med[(med["framework"] == fw) & (med["problem"] == prob)]
            vals.append(float(row["peak_memory_mb"].iloc[0]) if len(row) else 0)
        offset = (i - (n_frameworks - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=fw, color=FRAMEWORK_COLORS[fw],
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Peak memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([PROBLEM_LABELS[p] for p in PROBLEM_ORDER])
    ax.legend(frameon=True, fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

"""
Convergence Plot Generator for VAMOS Paper
===========================================
Creates a multi-panel figure showing HV convergence trajectories for VAMOS
(Numba) vs pymoo on representative benchmark problems.

Usage: python paper/19_plot_convergence.py

Reads:
  - experiments/convergence_paper.csv

Writes:
  - paper/manuscript/figures/convergence.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
INPUT_CSV = ROOT_DIR / "experiments" / "convergence_paper.csv"
OUTPUT_FIG = Path(__file__).parent / "manuscript" / "figures" / "convergence.png"

# Display settings
FRAMEWORK_STYLES = {
    "VAMOS (numba)": {"color": "#1f77b4", "label": "VAMOS (Numba)", "ls": "-"},
    "pymoo": {"color": "#ff7f0e", "label": "pymoo", "ls": "--"},
}

PROBLEM_TITLES = {
    "zdt1": "ZDT1 (2-obj, 30 vars)",
    "dtlz2": "DTLZ2 (3-obj, 12 vars)",
    "wfg4": "WFG4 (2-obj, 24 vars)",
}


def plot_convergence(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a multi-panel convergence figure."""
    problems = df["problem"].unique()
    n_panels = len(problems)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5), squeeze=False)
    axes = axes.flatten()

    for i, problem in enumerate(sorted(problems)):
        ax = axes[i]
        problem_data = df[df["problem"] == problem]

        for fw, style in FRAMEWORK_STYLES.items():
            fw_data = problem_data[problem_data["framework"] == fw]
            if fw_data.empty:
                continue

            # Pivot to get seeds as rows and n_evals as columns
            pivot = fw_data.pivot_table(
                index="seed", columns="n_evals", values="hypervolume", aggfunc="first"
            )
            evals = np.array(pivot.columns, dtype=float)

            # Compute median and IQR across seeds
            medians = pivot.median(axis=0).values
            q25 = pivot.quantile(0.25, axis=0).values
            q75 = pivot.quantile(0.75, axis=0).values

            ax.plot(evals / 1000, medians, color=style["color"], ls=style["ls"],
                    label=style["label"], linewidth=1.5)
            ax.fill_between(evals / 1000, q25, q75, color=style["color"], alpha=0.15)

        title = PROBLEM_TITLES.get(problem, problem.upper())
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Evaluations (x1000)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Normalized HV", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(FRAMEWORK_STYLES),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence figure to {output_path}")


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run 18_run_convergence_experiment.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Problems: {df['problem'].unique().tolist()}")
    print(f"Frameworks: {df['framework'].unique().tolist()}")
    print(f"Seeds: {df['seed'].nunique()}")

    plot_convergence(df, OUTPUT_FIG)


if __name__ == "__main__":
    main()

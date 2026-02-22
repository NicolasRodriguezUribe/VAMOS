"""
Generate forest plots for per-problem HV equivalence CI intervals.

Each plot shows horizontal CI bars for each problem, with a shaded +-1%
equivalence margin band and a vertical dashed line at 0.

Usage: python paper/27_plot_forest_ci.py
       python paper/27_plot_forest_ci.py --algo nsgaii

Reads:  experiments/ci_details_{algo}.csv
Writes: paper/manuscript/figures/forest_ci_{algo}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "experiments"
FIG_DIR = Path(__file__).parent / "manuscript" / "figures"

ALGO_SPECS = {
    "nsgaii": {"display": "NSGA-II", "csv": "ci_details_nsgaii.csv"},
    "smsemoa": {"display": "SMS-EMOA", "csv": "ci_details_smsemoa.csv"},
    "moead": {"display": "MOEA/D", "csv": "ci_details_moead.csv"},
}

# Problem order grouped by family
PROBLEM_ORDER = [
    "zdt1", "zdt2", "zdt3", "zdt4", "zdt6",
    "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7",
    "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9",
]

FRAMEWORK_COLORS = {"pymoo": "#DD8452", "jMetalPy": "#55A868"}
EQUIV_MARGIN = 1.0  # +/-1% equivalence margin
CLIP_RANGE = 5.0    # clip axis to +/-5%


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forest plots for HV equivalence CI.")
    parser.add_argument("--algo", action="append", choices=sorted(ALGO_SPECS.keys()),
                        help="Algorithm(s) to plot. Default: all.")
    return parser.parse_args()


def plot_forest(algo_key: str, spec: dict) -> None:
    csv_path = DATA_DIR / spec["csv"]
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    frameworks = [fw for fw in ["pymoo", "jMetalPy"] if fw in df["framework"].unique()]
    if not frameworks:
        print(f"  [SKIP] No competitor frameworks in {csv_path}")
        return

    # Filter to known problems
    df = df[df["problem"].isin(PROBLEM_ORDER)]
    problems = [p for p in PROBLEM_ORDER if p in df["problem"].unique()]
    n_problems = len(problems)

    fig, ax = plt.subplots(figsize=(3.4, 0.22 * n_problems + 0.9))

    y_offset = {"pymoo": 0.15, "jMetalPy": -0.15}
    if len(frameworks) == 1:
        y_offset = {frameworks[0]: 0.0}

    # Equivalence margin band
    ax.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color="lightgray", alpha=0.35, zorder=0)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)

    for fw in frameworks:
        fw_data = df[df["framework"] == fw]
        color = FRAMEWORK_COLORS[fw]
        offset = y_offset[fw]

        for i, prob in enumerate(problems):
            row = fw_data[fw_data["problem"] == prob]
            if row.empty:
                continue
            lo = float(row["ci_low_percent"].iloc[0])
            hi = float(row["ci_high_percent"].iloc[0])
            med = float(row["median_delta_percent"].iloc[0])
            y = n_problems - 1 - i + offset

            # Clip for display
            lo_clipped = max(lo, -CLIP_RANGE)
            hi_clipped = min(hi, CLIP_RANGE)

            # Draw CI bar
            ax.plot([lo_clipped, hi_clipped], [y, y], color=color, linewidth=1.8,
                    solid_capstyle="butt", zorder=2)

            # Draw median dot
            med_clipped = max(min(med, CLIP_RANGE), -CLIP_RANGE)
            ax.plot(med_clipped, y, "o", color=color, markersize=2.5, zorder=3)

            # Arrow indicators for clipped CIs
            if lo < -CLIP_RANGE:
                ax.annotate("", xy=(-CLIP_RANGE - 0.15, y), xytext=(-CLIP_RANGE + 0.15, y),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
            if hi > CLIP_RANGE:
                ax.annotate("", xy=(CLIP_RANGE + 0.15, y), xytext=(CLIP_RANGE - 0.15, y),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # Family dividers
    divider_positions = [
        n_problems - 5 - 0.5,   # between zdt6 and dtlz1
        n_problems - 12 - 0.5,  # between dtlz7 and wfg1
    ]
    for ypos in divider_positions:
        ax.axhline(ypos, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    # Y-axis
    ax.set_yticks(range(n_problems))
    ax.set_yticklabels([p.upper() for p in reversed(problems)], fontsize=6)
    ax.set_ylim(-0.5, n_problems - 0.5)

    # X-axis
    ax.set_xlim(-CLIP_RANGE - 0.5, CLIP_RANGE + 0.5)
    ax.set_xlabel(r"$\Delta$HV (%)", fontsize=7.5)
    ax.set_title(f"{spec['display']} — Equivalence CI (90%)", fontsize=8)

    # Legend
    handles = []
    for fw in frameworks:
        handles.append(mpatches.Patch(color=FRAMEWORK_COLORS[fw], label=f"vs {fw}"))
    handles.append(mpatches.Patch(color="lightgray", alpha=0.5, label=r"$\pm$1% margin"))
    ax.legend(handles=handles, fontsize=6, loc="lower right", frameon=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"forest_ci_{algo_key}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    args = _parse_args()
    algos = args.algo or list(ALGO_SPECS.keys())
    for algo_key in algos:
        print(f"Plotting {algo_key}...")
        plot_forest(algo_key, ALGO_SPECS[algo_key])


if __name__ == "__main__":
    main()

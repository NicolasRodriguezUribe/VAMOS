"""
Generate a dual-panel runtime heatmap figure.

Left panel:  VAMOS backend comparison (Numba, MooCore, NumPy)
Right panel: Framework comparison (VAMOS, pymoo, jMetalPy)

Usage: python paper/28_plot_runtime_heatmap.py

Reads:  experiments/benchmark_paper.csv
Writes: paper/generated/figures/runtime_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "experiments"
FIG_DIR = Path(__file__).parent / "generated" / "figures"

INPUT_CSV = DATA_DIR / "benchmark_paper.csv"
OUTPUT_PNG = FIG_DIR / "runtime_heatmap.png"

# Problem display order grouped by family
PROBLEM_ORDER = [
    "zdt1", "zdt2", "zdt3", "zdt4", "zdt6",
    "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7",
    "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9",
]
PROBLEM_LABELS = {p: p.upper() for p in PROBLEM_ORDER}

# Family divider positions (index of first problem in each family)
FAMILY_DIVIDERS = [5, 12]  # after zdt6 (idx 4), after dtlz7 (idx 11)

BACKEND_ORDER = ["numba", "moocore", "numpy"]
BACKEND_LABELS = {"numba": "Numba", "moocore": "MooCore", "numpy": "NumPy"}

FRAMEWORK_ORDER = ["VAMOS", "pymoo", "jMetalPy"]


def _get_family(problem: str) -> str:
    if problem.startswith("zdt"):
        return "ZDT"
    if problem.startswith("dtlz"):
        return "DTLZ"
    if problem.startswith("wfg"):
        return "WFG"
    return "Other"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the manuscript runtime heatmap.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated figure.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing figure: {args.output}. Pass --overwrite to replace it.")
    df = pd.read_csv(INPUT_CSV)

    # Compute per-problem median runtime
    medians = df.groupby(["framework", "problem"])["runtime_seconds"].median()

    # Build backend matrix
    backend_data = np.full((len(PROBLEM_ORDER), len(BACKEND_ORDER)), np.nan)
    for j, backend in enumerate(BACKEND_ORDER):
        fw_name = f"VAMOS ({backend})"
        for i, prob in enumerate(PROBLEM_ORDER):
            try:
                backend_data[i, j] = medians.loc[(fw_name, prob)]
            except KeyError:
                pass

    # Build framework matrix
    framework_data = np.full((len(PROBLEM_ORDER), len(FRAMEWORK_ORDER)), np.nan)
    for j, fw in enumerate(FRAMEWORK_ORDER):
        fw_lookup = "VAMOS (numba)" if fw == "VAMOS" else fw
        for i, prob in enumerate(PROBLEM_ORDER):
            try:
                framework_data[i, j] = medians.loc[(fw_lookup, prob)]
            except KeyError:
                pass

    # Color normalization (log scale, shared across both panels)
    all_vals = np.concatenate([backend_data.ravel(), framework_data.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = max(all_vals.min() * 0.8, 0.1)
    vmax = all_vals.max() * 1.2
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 7.5), sharey=True,
                                    gridspec_kw={"width_ratios": [len(BACKEND_ORDER), len(FRAMEWORK_ORDER)],
                                                 "wspace": 0.08})

    def _draw_heatmap(ax, data, col_labels, title):
        n_rows, n_cols = data.shape
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                        interpolation="nearest")

        # Annotate cells
        for i in range(n_rows):
            row_vals = data[i, :]
            row_valid = row_vals[np.isfinite(row_vals)]
            row_min = row_valid.min() if len(row_valid) > 0 else np.inf
            for j in range(n_cols):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color="gray")
                    continue
                is_best = abs(val - row_min) < 1e-9
                weight = "bold" if is_best else "normal"
                # Choose text color based on background brightness
                bg_color = cmap(norm(val))
                brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = "white" if brightness < 0.5 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, fontweight=weight, color=text_color)

        # Family divider lines
        for div in FAMILY_DIVIDERS:
            ax.axhline(y=div - 0.5, color="white", linewidth=2)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        ax.tick_params(left=False, bottom=False)

        # Remove frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        return im

    # Left panel: backends
    _draw_heatmap(ax1, backend_data,
                  [BACKEND_LABELS[b] for b in BACKEND_ORDER],
                  "VAMOS Backends")

    # Right panel: frameworks
    im = _draw_heatmap(ax2, framework_data,
                       FRAMEWORK_ORDER,
                       "Cross-framework")

    # Y-axis labels (on left panel only)
    ax1.set_yticks(range(len(PROBLEM_ORDER)))
    ax1.set_yticklabels([PROBLEM_LABELS[p] for p in PROBLEM_ORDER], fontsize=8)

    # Family labels on the far left
    family_ranges = [(0, 4, "ZDT"), (5, 11, "DTLZ"), (12, 20, "WFG")]
    for start, end, label in family_ranges:
        y_center = (start + end) / 2
        ax1.text(-0.8, y_center, label, ha="center", va="center",
                 fontsize=9, fontweight="bold", rotation=90,
                 transform=ax1.get_yaxis_transform())

    # Shared colorbar
    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("Median runtime (seconds)", fontsize=9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

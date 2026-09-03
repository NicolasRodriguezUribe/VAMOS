"""
Pareto Front Variant Experiment & Plot Generator
=================================================
Runs four NSGA-II variants on ZDT4 and DTLZ2, saves obtained fronts as CSV,
and generates publication-quality Pareto front plots.

Variants:
  1. ZDT4  – Standard NSGA-II          (30 000 evaluations)
  2. ZDT4  – Steady-State NSGA-II      (30 000 evaluations)
  3. DTLZ2 – Standard NSGA-II          (40 000 evaluations)
  4. DTLZ2 – Unbounded-Archive NSGA-II (40 000 evaluations)

Usage: python paper/32_run_pareto_front_variants.py

Writes:
  - paper/manuscript/scripts/pareto_front_plots/data/FUN.NSGAII.ZDT4.csv
  - paper/manuscript/scripts/pareto_front_plots/data/FUN.NSGAII.ZDT4.steady_state.csv
  - paper/manuscript/scripts/pareto_front_plots/data/FUN.NSGAII.DTLZ2.csv
  - paper/manuscript/scripts/pareto_front_plots/data/FUN.NSGAII.DTLZ2.archive.csv
  - paper/generated/figures/front_zdt4_standard.png
  - paper/generated/figures/front_zdt4_steady_state.png
  - paper/generated/figures/front_dtlz2_standard.png
  - paper/generated/figures/front_dtlz2_archive.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.components.subset_selection import select_top_k_farthest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "manuscript" / "scripts" / "pareto_front_plots" / "data"
FIG_DIR = Path(__file__).parent / "generated" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Problem configs
# ---------------------------------------------------------------------------
SEED = 42
POP_SIZE = 100
# ZDT4's multimodality makes the fixed-seed 20k runs occasionally miss the
# right-hand extreme; 30k yields a representative front for both variants.
ZDT4_PLOT_EVALS = 30_000

EXPERIMENTS = [
    {
        "problem": "zdt4",
        "n_var": 10,
        "n_obj": 2,
        "n_evals": ZDT4_PLOT_EVALS,
        "variant": "standard",
        "csv_name": "FUN.NSGAII.ZDT4.csv",
        "fig_name": "front_zdt4_standard.png",
        "title": "ZDT4 \u2014 Standard NSGA-II",
    },
    {
        "problem": "zdt4",
        "n_var": 10,
        "n_obj": 2,
        "n_evals": ZDT4_PLOT_EVALS,
        "variant": "steady_state",
        "csv_name": "FUN.NSGAII.ZDT4.steady_state.csv",
        "fig_name": "front_zdt4_steady_state.png",
        "title": "ZDT4 \u2014 Steady-State NSGA-II",
    },
    {
        "problem": "dtlz2",
        "n_var": 12,
        "n_obj": 3,
        "n_evals": 40_000,
        "variant": "standard",
        "csv_name": "FUN.NSGAII.DTLZ2.csv",
        "fig_name": "front_dtlz2_standard.png",
        "title": "DTLZ2 \u2014 Standard NSGA-II",
    },
    {
        "problem": "dtlz2",
        "n_var": 12,
        "n_obj": 3,
        "n_evals": 40_000,
        "variant": "archive",
        "csv_name": "FUN.NSGAII.DTLZ2.archive.csv",
        "fig_name": "front_dtlz2_archive.png",
        "title": "DTLZ2 \u2014 Archive NSGA-II",
    },
]

# Reference front files (headerless CSV)
REF_FRONTS = {
    "zdt4": "ZDT4.csv",
    "dtlz2": "DTLZ2.3D.csv",
}


# ---------------------------------------------------------------------------
# Build NSGA-II configs
# ---------------------------------------------------------------------------

def _build_config(n_var: int, variant: str) -> NSGAIIConfig:
    builder = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=1.0 / n_var, eta=20.0)
        .selection("tournament")
    )
    if variant == "steady_state":
        builder = builder.offspring_size(1)
    elif variant == "archive":
        builder = builder.external_archive(capacity=None)
    return builder.build()


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(exp: dict) -> np.ndarray:
    """Run a single NSGA-II variant and return the obtained objective front."""
    cfg = _build_config(exp["n_var"], exp["variant"])
    print(f"  Running {exp['title']} ({exp['n_evals']} evals) ...")

    result = optimize(
        exp["problem"],
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", exp["n_evals"]),
        seed=SEED,
        engine="numba",
    )

    F = result.F

    # For archive variant, use farthest-point sampling for uniform spread
    if exp["variant"] == "archive":
        archive_payload = result.data.get("archive") if hasattr(result, "data") else None
        if isinstance(archive_payload, dict):
            archive_F = archive_payload.get("F")
            if archive_F is not None:
                if archive_F.shape[0] > POP_SIZE:
                    idx = select_top_k_farthest(archive_F, POP_SIZE)
                    F = archive_F[idx]
                else:
                    F = archive_F

    return F


# ---------------------------------------------------------------------------
# Plotting (mirrors plot_fronts.py style)
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


REF_COLOR = "#1f77b4"      # blue for reference front
FRONT_COLOR = "#E74C3C"    # red for obtained solutions


def plot_2d(ref: np.ndarray, obt: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    idx = np.argsort(ref[:, 0])
    ax.plot(
        ref[idx, 0], ref[idx, 1],
        color=REF_COLOR, linewidth=1.2, label="Reference front", zorder=2,
    )
    ax.scatter(
        obt[:, 0], obt[:, 1],
        s=9, color=FRONT_COLOR, edgecolors="white", linewidths=0.2,
        alpha=0.85, label="NSGA-II", zorder=3,
    )

    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(out_path)
    plt.close(fig)


def plot_3d(ref: np.ndarray, obt: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        ref[:, 0], ref[:, 1], ref[:, 2],
        s=1, color=REF_COLOR, alpha=0.15, label="Reference front", zorder=1,
    )
    ax.scatter(
        obt[:, 0], obt[:, 1], obt[:, 2],
        s=10, color=FRONT_COLOR, edgecolors="white", linewidths=0.2,
        alpha=0.9, label="NSGA-II", zorder=3,
    )

    ax.set_xlabel("$f_1$", labelpad=4)
    ax.set_ylabel("$f_2$", labelpad=4)
    ax.set_zlabel("$f_3$", labelpad=4)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 0.95), framealpha=0.9)
    ax.view_init(elev=35, azim=45)
    ax.tick_params(axis="both", labelsize=7)

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and plot the manuscript Pareto-front variants.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated CSV/PNG outputs.")
    return parser.parse_args()


def main() -> None:
    import matplotlib
    args = _parse_args()
    outputs = [DATA_DIR / exp["csv_name"] for exp in EXPERIMENTS] + [FIG_DIR / exp["fig_name"] for exp in EXPERIMENTS]
    collisions = [path for path in outputs if path.exists()]
    if collisions and not args.overwrite:
        names = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing Pareto output(s): {names}. Pass --overwrite to replace them.")
    matplotlib.use("Agg")
    _setup_style()

    print("Running Pareto front variant experiments ...")

    for exp in EXPERIMENTS:
        # 1. Run optimization
        F = run_experiment(exp)

        # 2. Save obtained front as headerless CSV
        csv_path = DATA_DIR / exp["csv_name"]
        np.savetxt(csv_path, F, delimiter=",", fmt="%.10f")
        print(f"    Saved {F.shape[0]} solutions -> {csv_path}")

        # 3. Load reference front
        ref_file = REF_FRONTS[exp["problem"]]
        ref = np.loadtxt(DATA_DIR / ref_file, delimiter=",")

        # 4. Generate figure
        fig_path = FIG_DIR / exp["fig_name"]
        if exp["n_obj"] == 2:
            plot_2d(ref, F, exp["title"], fig_path)
        else:
            plot_3d(ref, F, exp["title"], fig_path)
        print(f"    Saved {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()

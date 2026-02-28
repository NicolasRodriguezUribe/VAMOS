"""Plot NSGA-II variant results from nsgaii_variants_archives.py on DTLZ2 (3-obj)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.components.archive import select_top_k_farthest
from vamos.problems import DTLZ2

ENGINE = os.environ.get("VAMOS_ADV_ENGINE", "numba")
MAX_EVALUATIONS = int(os.environ.get("VAMOS_ADV_EVALS", "40000"))
SEED = int(os.environ.get("VAMOS_ADV_SEED", "11"))
POP_SIZE = 100

# Reference front
REF_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "manuscript" / "scripts" / "pareto_front_plots" / "data"
REF_FILE = REF_DIR / "DTLZ2.3D.csv"

OUTPUT = Path(__file__).resolve().parent / "nsgaii_variants_dtlz2.png"

# Colors
REF_COLOR = "#1f77b4"
COLORS = {
    "Generational": "#E74C3C",
    "Steady-State": "#2ca02c",
    "Bounded Archive": "#ff7f0e",
    "Unbounded Archive": "#9467bd",
}


def build_configs():
    base = lambda: (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )
    return {
        "Generational": base().offspring_size(POP_SIZE).build(),
        "Steady-State": base().offspring_size(1).build(),
        "Bounded Archive": base().offspring_size(POP_SIZE).external_archive(capacity=100, pruning="crowding").build(),
        "Unbounded Archive": base().offspring_size(POP_SIZE).external_archive().build(),
    }


def get_front(result, variant_name):
    """Extract the objective front, using archive if available."""
    F = result.F
    if "Archive" in variant_name:
        archive = result.data.get("archive")
        if isinstance(archive, dict) and archive.get("F") is not None:
            archive_F = np.asarray(archive["F"], dtype=float)
            if variant_name == "Unbounded Archive":
                idx = select_top_k_farthest(archive_F, POP_SIZE)
                F = archive_F[idx]
            else:
                F = archive_F
    return F


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    problem = DTLZ2(n_obj=3)
    configs = build_configs()

    # Load reference front
    ref = np.loadtxt(REF_FILE, delimiter=",") if REF_FILE.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), subplot_kw={"projection": "3d"})

    for ax, (name, cfg) in zip(axes.flatten(), configs.items()):
        print(f"Running {name} ...")
        result = optimize(
            problem, algorithm="nsgaii", algorithm_config=cfg,
            max_evaluations=MAX_EVALUATIONS, seed=SEED, engine=ENGINE,
        )
        F = get_front(result, name)

        # Reference
        if ref is not None:
            ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2],
                       s=1, color=REF_COLOR, alpha=0.10, zorder=1)

        # Obtained
        ax.scatter(F[:, 0], F[:, 1], F[:, 2],
                   s=12, color=COLORS[name], edgecolors="white", linewidths=0.2,
                   alpha=0.9, zorder=3)

        ax.set_title(name, fontsize=10, pad=8)
        ax.set_xlabel("$f_1$", labelpad=2, fontsize=8)
        ax.set_ylabel("$f_2$", labelpad=2, fontsize=8)
        ax.set_zlabel("$f_3$", labelpad=2, fontsize=8)
        ax.view_init(elev=35, azim=45)
        ax.tick_params(axis="both", labelsize=6)

        n_sol = F.shape[0]
        archive = result.data.get("archive")
        archive_size = np.asarray(archive["F"]).shape[0] if isinstance(archive, dict) and archive.get("F") is not None else 0
        label = f"{n_sol} solutions"
        if archive_size:
            label += f" (archive: {archive_size})"
        ax.text2D(0.02, 0.02, label, transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

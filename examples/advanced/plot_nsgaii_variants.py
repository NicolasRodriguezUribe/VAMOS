"""
Plot NSGA-II variant results on DTLZ2 without depending on paper assets.

Usage:
    python examples/advanced/plot_nsgaii_variants.py
    python examples/advanced/plot_nsgaii_variants.py --output results/examples/nsgaii_variants_dtlz2.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.components.subset_selection import select_top_k_farthest
from vamos.problems import DTLZ2


ENGINE = os.environ.get("VAMOS_ADV_ENGINE", "numba")
MAX_EVALUATIONS = int(os.environ.get("VAMOS_ADV_EVALS", "40000"))
SEED = int(os.environ.get("VAMOS_ADV_SEED", "11"))
POP_SIZE = 100
DEFAULT_OUTPUT = Path("results/examples/nsgaii_variants_dtlz2.png")

COLORS = {
    "Generational": "#D04A3A",
    "Steady-State": "#2E8B57",
    "Bounded Archive": "#D9822B",
    "Unbounded Archive": "#5D6FB8",
}


def _build_configs() -> dict[str, NSGAIIConfig]:
    def _base():
        return (
            NSGAIIConfig.builder()
            .pop_size(POP_SIZE)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("polynomial", prob="1/n", eta=20.0)
            .selection("tournament", size=2)
        )

    return {
        "Generational": _base().offspring_size(POP_SIZE).build(),
        "Steady-State": _base().offspring_size(1).build(),
        "Bounded Archive": _base().offspring_size(POP_SIZE).external_archive(capacity=100, pruning="crowding").build(),
        "Unbounded Archive": _base().offspring_size(POP_SIZE).external_archive().build(),
    }


def _sample_reference_front(n: int = 800) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(3), size=n)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _extract_front(result, variant_name: str) -> np.ndarray:
    front = np.asarray(result.F, dtype=float)
    archive = result.data.get("archive")
    if "Archive" not in variant_name or not isinstance(archive, dict) or archive.get("F") is None:
        return front

    archive_f = np.asarray(archive["F"], dtype=float)
    if variant_name == "Unbounded Archive" and archive_f.shape[0] > POP_SIZE:
        idx = select_top_k_farthest(archive_f, POP_SIZE)
        return archive_f[idx]
    return archive_f


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NSGA-II variant fronts on DTLZ2.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the PNG figure.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    problem = DTLZ2(n_obj=3)
    reference = _sample_reference_front()
    configs = _build_configs()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), subplot_kw={"projection": "3d"})

    for ax, (name, cfg) in zip(axes.flatten(), configs.items()):
        print(f"Running {name} ...")
        result = optimize(
            problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            max_evaluations=MAX_EVALUATIONS,
            seed=SEED,
            engine=ENGINE,
        )
        front = _extract_front(result, name)

        ax.scatter(reference[:, 0], reference[:, 1], reference[:, 2], s=1, color="#1f77b4", alpha=0.08, zorder=1)
        ax.scatter(
            front[:, 0],
            front[:, 1],
            front[:, 2],
            s=12,
            color=COLORS[name],
            edgecolors="white",
            linewidths=0.2,
            alpha=0.9,
            zorder=3,
        )

        ax.set_title(name, fontsize=10, pad=8)
        ax.set_xlabel("$f_1$", labelpad=2, fontsize=8)
        ax.set_ylabel("$f_2$", labelpad=2, fontsize=8)
        ax.set_zlabel("$f_3$", labelpad=2, fontsize=8)
        ax.view_init(elev=35, azim=45)
        ax.tick_params(axis="both", labelsize=6)

        archive = result.data.get("archive")
        archive_size = np.asarray(archive["F"]).shape[0] if isinstance(archive, dict) and archive.get("F") is not None else 0
        label = f"{front.shape[0]} solutions"
        if archive_size:
            label += f" (archive: {archive_size})"
        ax.text2D(0.02, 0.02, label, transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

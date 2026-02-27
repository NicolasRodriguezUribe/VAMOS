"""
External archive demo with NSGA-II on DTLZ2 (3 objectives).

Shows how to configure a hypervolume/crowding archive, access its contents,
select the top-k most spread solutions via crowding distance, and plot the
archived Pareto set alongside the final population.

Note: This example uses explicit config objects for advanced control.
For quick runs, prefer the unified optimize(...) API.

Usage:
    python examples/advanced/archive_usage_nsga2.py

Requirements:
    pip install -e ".[examples]"  # matplotlib for plotting
"""

from __future__ import annotations

import numpy as np

from vamos import optimize
from vamos.problems import DTLZ2
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.components.archive import select_top_k_crowding
from vamos.engine.archive import BoundedArchive, BoundedArchiveConfig


def build_config() -> NSGAIIConfig:
    """
    Configure NSGA-II with an external archive.

    DTLZ2 is 3-objective here, so we keep a standard NSGA-II setup
    and enable an external archive for additional non-dominated points.
    """
    return (
        NSGAIIConfig.builder()
        .pop_size(80)
        .offspring_size(80)
        .crossover("")
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .external_archive()
        .build()
    )


def main() -> None:
    problem = DTLZ2(n_obj=3)
    cfg = build_config()

    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        max_evaluations=40000,
        seed=11,
        engine="numpy",
    )

    F = result.F
    archive = result.data.get("archive") or {}
    archive_F = archive.get("F")
    archive_X = archive.get("X")

    print(f"Population size: {len(F)}")
    if archive_F is not None:
        print(f"Archive size: {len(archive_F)}")

        # Select the top-80 most spread solutions using crowding distance.
        top_idx = select_top_k_crowding(archive_F, k=80)
        top_F = archive_F[top_idx]
        print(f"Top-{len(top_idx)} spread solutions selected from archive")
        best_idx = int(np.argmin(np.sum(top_F[:, :3], axis=1)))
        print("Best archived objectives (sum-min heuristic):", top_F[best_idx])
    else:
        print("No archive returned. Check external_archive config.")

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c="lightgray", label="Final population", alpha=0.6)
        if archive_F is not None:
            ax.scatter(archive_F[:, 0], archive_F[:, 1], archive_F[:, 2], c="crimson", label="External archive", alpha=0.75)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title("NSGA-II with external archive on DTLZ2")
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print("Plotting skipped:", exc)

    if archive_X is not None and archive_F is not None:
        print("Archive snapshot (first solution):")
        print("  X:", archive_X[0])
        print("  F:", archive_F[0])


def _make_dense_front(n_points: int = 40) -> np.ndarray:
    # Synthetic 2D non-dominated front with many near-duplicate trade-offs.
    t = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([t, 1.0 - t])


def _demo_prune_policy_matters() -> None:
    # With the simplified archive config, only prune policy affects bounded behavior.
    F = _make_dense_front(40)
    cfg_crowding = BoundedArchiveConfig(
        size_cap=8,
        prune_policy="crowding",
    )
    cfg_random = BoundedArchiveConfig(
        size_cap=8,
        prune_policy="random",
    )

    arc_crowding = BoundedArchive(cfg_crowding)
    upd_crowding = arc_crowding.add(X=None, F=F, evals=F.shape[0])
    arc_random = BoundedArchive(cfg_random)
    upd_random = arc_random.add(X=None, F=F, evals=F.shape[0])

    print("\n[prune_policy_demo] Same size_cap, different prune policy")
    print(f"  crowding: final={upd_crowding.after}, reason={upd_crowding.prune_reason}")
    print(f"  random:   final={upd_random.after}, reason={upd_random.prune_reason}")


if __name__ == "__main__":
    main()
    _demo_prune_policy_matters()

"""
External archive demo with NSGA-II on DTLZ2 (3 objectives).

Shows how to configure a hypervolume/crowding archive, access its contents,
and plot the archived Pareto set alongside the final population.

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
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
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
        best_idx = int(np.argmin(np.sum(archive_F[:, :3], axis=1)))
        print("Best archived objectives (sum-min heuristic):", archive_F[best_idx])
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


def _grid_cells(F: np.ndarray, epsilon: float) -> int:
    keys = np.floor(F / float(epsilon)).astype(int)
    return int(np.unique(keys, axis=0).shape[0])


def _demo_archive_type_matters() -> None:
    # Here archive_type matters because BoundedArchive applies different
    # reduction paths: plain size-cap pruning vs epsilon-grid compaction first.
    F = _make_dense_front(40)
    cfg_size_cap = BoundedArchiveConfig(
        size_cap=8,
        archive_type="size_cap",
        prune_policy="crowding",
        epsilon=0.5,
    )
    cfg_epsilon = BoundedArchiveConfig(
        size_cap=8,
        archive_type="epsilon_grid",
        prune_policy="crowding",
        epsilon=0.5,
    )

    arc_size_cap = BoundedArchive(cfg_size_cap)
    upd_size_cap = arc_size_cap.add(X=None, F=F, evals=F.shape[0])
    arc_epsilon = BoundedArchive(cfg_epsilon)
    upd_epsilon = arc_epsilon.add(X=None, F=F, evals=F.shape[0])

    print("\n[archive_type_demo] Same size_cap/prune_policy, different archive_type")
    print(
        f"  size_cap: final={upd_size_cap.after}, reason={upd_size_cap.prune_reason}, "
        f"cells={_grid_cells(arc_size_cap.F, cfg_size_cap.epsilon)}"
    )
    print(
        f"  epsilon_grid: final={upd_epsilon.after}, reason={upd_epsilon.prune_reason}, "
        f"cells={_grid_cells(arc_epsilon.F, cfg_epsilon.epsilon)}"
    )
    print("  Why: epsilon_grid compacts objective-space cells before crowding prune.")


if __name__ == "__main__":
    main()
    _demo_archive_type_matters()

"""
External archive demo with NSGA-II on ZDT1.

Shows how to configure a hypervolume/crowding archive, access its contents,
and plot the archived Pareto set alongside the final population.

Usage:
    python examples/archive_usage_nsga2.py

Requirements:
    pip install -e ".[examples]"  # matplotlib for plotting
"""
from __future__ import annotations

import numpy as np

from vamos.api import OptimizeConfig, optimize
from vamos.foundation.problems_registry import ZDT1
from vamos.engine.api import NSGAIIConfig


def build_config(archive_type: str = "hypervolume") -> NSGAIIConfig:
    """
    Configure NSGA-II with an external archive.

    archive_type: "hypervolume" (default, prefers high HV) or "crowding" (spread)
    """
    return (
        NSGAIIConfig()
        .pop_size(80)
        .offspring_size(80)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        
        .engine("numpy")
        .external_archive(size=100, archive_type=archive_type)
        .fixed()
    )


def main() -> None:
    problem = ZDT1(n_var=30)
    cfg = build_config(archive_type="hypervolume")

    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 8000),
            seed=11,
            engine="numpy",
        )
    )

    F = result.F
    archive = result.data.get("archive") or {}
    archive_F = archive.get("F")
    archive_X = archive.get("X")

    print(f"Population size: {len(F)}")
    if archive_F is not None:
        print(f"Archive size ({cfg.archive_type or 'hypervolume'}): {len(archive_F)}")
        hv_best_idx = int(np.argmin(archive_F[:, 0] + archive_F[:, 1]))
        print("Best archived objectives (sum-min heuristic):", archive_F[hv_best_idx])
    else:
        print("No archive returned. Check archive_size and archive_type.")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(F[:, 0], F[:, 1], c="lightgray", label="Final population", alpha=0.6)
        if archive_F is not None:
            plt.scatter(archive_F[:, 0], archive_F[:, 1], c="crimson", label="External archive", alpha=0.85)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("NSGA-II with external archive on ZDT1")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional
        print("Plotting skipped:", exc)

    if archive_X is not None and archive_F is not None:
        print("Archive snapshot (first solution):")
        print("  X:", archive_X[0])
        print("  F:", archive_F[0])


if __name__ == "__main__":
    main()

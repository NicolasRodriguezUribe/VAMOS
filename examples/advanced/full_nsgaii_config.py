"""
NSGA-II with every builder parameter.

Demonstrates every configurable option available on NSGAIIConfig.builder():
  pop_size, offspring_size, crossover, mutation, selection, initializer,
  repair, external_archive, constraint_mode, result_mode, mutation_prob_factor,
  track_genealogy, live_callback_mode, generation_callback,
  adaptive_operator_selection, immigration, and parent_selection_filter.

Usage:
    python examples/advanced/full_nsgaii_config.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig


def _generation_logger(gen: int, pop: dict[str, Any]) -> None:
    """Example generation callback: print progress every 50 generations."""
    if gen % 50 == 0:
        F = np.asarray(pop["F"], dtype=float)
        print(f"  gen {gen:>4d}  |  pop {F.shape[0]}  |  f1 min={F[:, 0].min():.4f}")


def main() -> None:
    config = (
        NSGAIIConfig.builder()
        # ── Population ───────────────────────────────────────────────
        .pop_size(100)  # Number of individuals
        .offspring_size(100)  # Offspring per generation (default: pop_size)
        # ── Operators ────────────────────────────────────────────────
        .crossover("sbx", prob=0.9, eta=20.0)  # Simulated Binary Crossover
        .mutation("polynomial", prob="1/n", eta=20.0)  # Polynomial Mutation (auto 1/n_var)
        .selection("tournament", size=2)  # Binary tournament selection
        # ── Initialization ───────────────────────────────────────────
        .initializer("sobol", scramble=True)  # Sobol quasi-random sequence
        # ── Repair ───────────────────────────────────────────────────
        .repair("clip")  # Clip out-of-bounds variables to bounds
        # ── Archive ──────────────────────────────────────────────────
        .external_archive(capacity=200, pruning="crowding")  # Bounded archive
        # ── Constraints ──────────────────────────────────────────────
        .constraint_mode("feasibility")  # Feasible-first ranking
        # ── Result ───────────────────────────────────────────────────
        .result_mode("non_dominated")  # Return only Pareto-optimal solutions
        # ── Mutation scaling ─────────────────────────────────────────
        .mutation_prob_factor(1.0)  # Multiplicative factor on mutation prob
        # ── Genealogy ────────────────────────────────────────────────
        .track_genealogy(True)  # Record parent-child relationships
        # ── Callbacks ────────────────────────────────────────────────
        .live_callback_mode("nd_only")  # Send only non-dominated to callbacks
        .generation_callback(_generation_logger, copy_arrays=True)
        # ── Advanced (optional) ──────────────────────────────────────
        .adaptive_operator_selection(None)  # No AOS (set dict to enable)
        .immigration(None)  # No immigration (set dict to enable)
        .parent_selection_filter(None)  # No parent filter (set callable to enable)
        .build()
    )

    print("Running NSGA-II (all parameters) on DTLZ2 ...")
    print(f"  config: {config}\n")

    result = optimize(
        "dtlz2",
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=40_000,
        seed=42,
        engine="numba",
    )

    F = result.F
    print(f"\nFound {len(F)} Pareto-optimal solutions")
    print(f"  f1: [{F[:, 0].min():.4f}, {F[:, 0].max():.4f}]")
    print(f"  f2: [{F[:, 1].min():.4f}, {F[:, 1].max():.4f}]")
    print(f"  f3: [{F[:, 2].min():.4f}, {F[:, 2].max():.4f}]")

    archive = result.data.get("archive")
    if isinstance(archive, dict) and archive.get("F") is not None:
        print(f"  Archive size: {np.asarray(archive['F']).shape[0]}")

    if config.track_genealogy:
        print("  Genealogy tracking: enabled")

    # ── Plotly 3D Pareto front ───────────────────────────────────────
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=F[:, 0],
                    y=F[:, 1],
                    z=F[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=F[:, 0], colorscale="Viridis", opacity=0.8),
                    name="Pareto front",
                )
            ]
        )
        fig.update_layout(
            title="NSGA-II on DTLZ2 — Pareto Front",
            scene=dict(xaxis_title="f1", yaxis_title="f2", zaxis_title="f3"),
            width=800,
            height=700,
        )
        fig.show()
    except ImportError:
        print("\nInstall plotly for visualization: pip install plotly")


if __name__ == "__main__":
    main()

"""
Tiny GCES vs NSGA-II comparison.

Runs both algorithms on one unconstrained problem with a fixed seed and small
budget. This is a lightweight smoke helper, not an experiment framework.

Usage:
    python examples/basics/gces_tiny_compare.py
"""

from __future__ import annotations

import numpy as np

from vamos import optimize


def _run(algo: str) -> None:
    result = optimize(
        "zdt1",
        algorithm=algo,
        pop_size=20,
        max_evaluations=80,
        seed=42,
        engine="numpy",
    )
    pop_F = np.asarray(result.data["population"]["F"])
    print(
        f"{algo}: pop_shape={pop_F.shape}, nd_shape={result.F.shape}, "
        f"finite={bool(np.isfinite(pop_F).all())}"
    )


def main() -> None:
    print("=== Tiny NSGA-II vs GCES smoke ===")
    _run("nsgaii")
    _run("gces")


if __name__ == "__main__":
    main()

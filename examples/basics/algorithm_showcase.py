"""
Algorithm showcase: AGE-MOEA and RVEA.

Demonstrates two alternative MOEA families beyond NSGA-II.
"""

from __future__ import annotations
from vamos import optimize
from vamos.algorithms import RVEAConfig


def run_agemoea():
    print("\n=== AGE-MOEA on ZDT1 ===")
    # AGE-MOEA is good for adaptive geometry estimation
    result = optimize("zdt1", algorithm="agemoea", pop_size=100, max_evaluations=10000, seed=1, verbose=True)
    print(f"AGE-MOEA found {len(result)} solutions")
    return result


def run_rvea():
    print("\n=== RVEA on DTLZ2 (3-obj) ===")
    # RVEA uses reference vectors, good for many-objective
    cfg = (
        RVEAConfig.builder()
        .pop_size(105)
        .n_partitions(13)  # H=13 for M=3 -> 105 refs
        .alpha(2.0)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .build()
    )
    result = optimize(
        "dtlz2",
        algorithm="rvea",
        algorithm_config=cfg,
        n_obj=3,
        max_evaluations=10000,
        seed=1,
        verbose=True,
    )
    print(f"RVEA found {len(result)} solutions")
    return result


if __name__ == "__main__":
    r1 = run_agemoea()
    r2 = run_rvea()

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        # Simple scatter for ZDT1
        plt.figure()
        plt.scatter(r1.F[:, 0], r1.F[:, 1], label="AGE-MOEA")
        plt.title("AGE-MOEA on ZDT1")
        plt.show()
    except ImportError:
        pass

"""
New Algorithms Demo: AGE-MOEA and RVEA.

Demonstrates the usage of the newly added algorithms.
"""

from __future__ import annotations
from vamos.api import optimize


def run_agemoea():
    print("\n=== AGE-MOEA on ZDT1 ===")
    # AGE-MOEA is good for adaptive geometry estimation
    result = optimize("zdt1", algorithm="agemoea", pop_size=100, budget=10000, seed=1, verbose=True)
    print(f"AGE-MOEA found {len(result)} solutions")
    return result


def run_rvea():
    print("\n=== RVEA on DTLZ2 (3-obj) ===")
    # RVEA uses reference vectors, good for many-objective
    result = optimize(
        "dtlz2",
        algorithm="rvea",
        n_obj=3,
        pop_size=105,  # H=14 for M=3 -> 105 refs
        budget=10000,
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

"""
Unified API Demonstration.

Showcases the power of `vamos.optimize()`:
1. AutoML mode (algorithm="auto")
2. Multi-seed execution (automatic list return)
3. Explicit configuration with specific engine
"""

from __future__ import annotations
from vamos import optimize


def demo_automl():
    print("\n=== 1. AutoML Mode ===")
    # Automatically selects algorithm based on problem traits (n_obj=2 -> NSGA-II usually)
    result = optimize("zdt1", algorithm="auto", budget=2000, verbose=True)
    print(f"Auto-selected result: {len(result)} solutions")


def demo_multiseed():
    print("\n=== 2. Multi-seed Study ===")
    # Passing a list of seeds triggers multi-run mode
    results = optimize(
        "zdt1",
        algorithm="nsgaii",
        budget=2000,
        seed=[0, 1, 2],  # 3 independent runs
        verbose=False,
    )
    print(f"Executed {len(results)} runs.")
    hv_vals = [len(r) for r in results]
    print(f"Solutions per run: {hv_vals}")


def demo_explicit():
    print("\n=== 3. Explicit Configuration ===")
    # Full control over hyperparameters
    result = optimize(
        "dtlz2",
        algorithm="moead",
        n_obj=3,  # Problem parameter
        pop_size=100,  # Algorithm parameter
        budget=5000,
        verbose=True,
    )
    print(f"MOEA/D result: {len(result)} solutions")


if __name__ == "__main__":
    demo_automl()
    demo_multiseed()
    demo_explicit()

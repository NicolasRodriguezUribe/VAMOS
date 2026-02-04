"""
Dask Distributed Example for VAMOS.

This example shows how to scale VAMOS optimization across a Dask cluster
for expensive objective function evaluations.

Requirements:
    pip install -e ".[compute]"

Usage:
    # Local cluster (for testing)
    python dask_cluster.py

    # Connect to existing cluster
    python dask_cluster.py --scheduler scheduler.example.com:8786
"""

from __future__ import annotations

import argparse
import time

import numpy as np


class ExpensiveProblem:
    """
    Simple ZDT1-like problem with an artificial per-individual delay.

    Notes:
        The evaluation backends (multiprocessing/dask) require the problem to be picklable.
        Keep the problem class at module scope (not nested in a function) for best compatibility.
    """

    encoding = "continuous"

    def __init__(self, n_var: int = 10, delay: float = 0.01) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.delay = float(delay)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        # Simulate an expensive evaluation cost per individual.
        time.sleep(self.delay * float(X.shape[0]))

        # ZDT1-like objectives
        f1 = X[:, 0]
        g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))

        out["F"] = np.column_stack([f1, f2])


def create_expensive_problem():
    return ExpensiveProblem()


def run_serial(problem, budget: int):
    """Run optimization without distributed evaluation."""
    import vamos

    print("Running SERIAL evaluation...")
    start = time.perf_counter()
    result = vamos.optimize(problem, algorithm="nsgaii", max_evaluations=budget, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")
    return elapsed


def run_distributed(problem, budget: int, scheduler: str | None = None):
    """Run optimization with Dask distributed evaluation."""
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        print('ERROR: Dask not installed. Run: pip install -e ".[compute]"')
        return None

    # Create or connect to cluster
    cluster = None
    if scheduler:
        print(f"Connecting to scheduler: {scheduler}")
        client = Client(scheduler)
    else:
        print("Creating local Dask cluster...")
        cluster = LocalCluster(n_workers=4, threads_per_worker=1)
        client = Client(cluster)

    print(f"  Dashboard: {client.dashboard_link}")

    # Import backend
    from vamos.foundation.eval.backends import DaskEvalBackend

    backend = DaskEvalBackend(client=client)

    print("Running DISTRIBUTED evaluation...")
    start = time.perf_counter()
    from vamos import optimize
    from vamos.algorithms import NSGAIIConfig

    algo_cfg = NSGAIIConfig.default(pop_size=50, n_var=problem.n_var)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=42,
        engine="numpy",
        eval_strategy=backend,
    )

    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")

    client.close()
    if cluster is not None:
        cluster.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Dask distributed VAMOS example")
    parser.add_argument("--scheduler", "-s", help="Dask scheduler address (e.g., scheduler:8786)")
    parser.add_argument("--budget", "-b", type=int, default=500, help="Evaluation budget (default: 500)")
    parser.add_argument("--delay", type=float, default=0.01, help="Artificial per-individual evaluation delay (seconds)")
    parser.add_argument("--compare", action="store_true", help="Compare serial vs distributed")
    args = parser.parse_args()

    problem = ExpensiveProblem(delay=args.delay)

    if args.compare:
        serial_time = run_serial(problem, args.budget)
        dist_time = run_distributed(problem, args.budget, args.scheduler)

        if dist_time and serial_time:
            speedup = serial_time / dist_time
            print(f"\nSpeedup: {speedup:.2f}x")
    else:
        run_distributed(problem, args.budget, args.scheduler)


if __name__ == "__main__":
    main()

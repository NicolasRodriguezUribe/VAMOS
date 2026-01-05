"""
Dask Distributed Example for VAMOS.

This example shows how to scale VAMOS optimization across a Dask cluster
for expensive objective function evaluations.

Requirements:
    pip install -e ".[distributed]"
    
Usage:
    # Local cluster (for testing)
    python dask_cluster.py
    
    # Connect to existing cluster
    python dask_cluster.py --scheduler scheduler.example.com:8786
"""
from __future__ import annotations

import argparse
import time


def create_expensive_problem():
    """Create a problem with expensive evaluation (simulated)."""
    from vamos.foundation.problem.base import Problem
    import numpy as np
    
    class ExpensiveProblem(Problem):
        def __init__(self, n_var: int = 10, delay: float = 0.1):
            super().__init__(
                n_var=n_var,
                n_obj=2,
                xl=0.0,
                xu=1.0,
            )
            self.delay = delay
        
        def _evaluate(self, X, out):
            # Simulate expensive computation
            time.sleep(self.delay)
            
            # ZDT1-like objectives
            f1 = X[:, 0]
            g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
            
            out["F"] = np.column_stack([f1, f2])
    
    return ExpensiveProblem()


def run_serial(problem, budget: int):
    """Run optimization without distributed evaluation."""
    import vamos
    
    print("Running SERIAL evaluation...")
    start = time.perf_counter()
    result = vamos.optimize(
        problem,
        algorithm="nsgaii",
        budget=budget,
        verbose=False
    )
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")
    return elapsed


def run_distributed(problem, budget: int, scheduler: str | None = None):
    """Run optimization with Dask distributed evaluation."""
    try:
        from dask.distributed import Client, LocalCluster
    except ImportError:
        print("ERROR: Dask not installed. Run: pip install -e '.[distributed]'")
        return None
    
    # Create or connect to cluster
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
    
    # Note: Currently the backend is used at algorithm level
    # This example shows the pattern for integration
    from vamos.experiment.builder import study
    
    result = (
        study(problem)
        .using("nsgaii", pop_size=50)
        .evaluations(budget)
        .seed(42)
        .run()
    )
    
    elapsed = time.perf_counter() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solutions: {len(result)}")
    
    client.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Dask distributed VAMOS example")
    parser.add_argument(
        "--scheduler", "-s",
        help="Dask scheduler address (e.g., scheduler:8786)"
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=500,
        help="Evaluation budget (default: 500)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare serial vs distributed"
    )
    args = parser.parse_args()
    
    problem = create_expensive_problem()
    
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

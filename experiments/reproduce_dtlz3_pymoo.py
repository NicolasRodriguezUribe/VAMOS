import sys
from pathlib import Path
import time
import numpy as np
import os

# Add relevant paths
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "paper"))

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from benchmark_utils import compute_hv, _reference_point, _reference_hv


def run_repro():
    problem_name = "dtlz3"
    seed = 0
    n_var = 12
    n_obj = 3
    pop_size = 100
    n_evals = 100000
    crossover_prob = 0.9
    crossover_eta = 20.0
    mutation_eta = 20.0

    print(f"Reproducing {problem_name} with pymoo, seed={seed}")

    pymoo_problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)

    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(prob=1.0 / n_var, eta=mutation_eta),
    )
    termination = get_termination("n_eval", n_evals)

    start = time.perf_counter()
    res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=True)
    elapsed = time.perf_counter() - start

    print(f"Runtime: {elapsed:.4f}s")

    if res.F is not None:
        print(f"Number of solutions: {len(res.F)}")
        print("First 5 solutions (F):")
        print(res.F[:5])
        print("Max objectives in F:")
        print(np.max(res.F, axis=0))
        print("Min objectives in F:")
        print(np.min(res.F, axis=0))

        # Check HV
        hv = compute_hv(res.F, problem_name)
        print(f"Computed HV: {hv}")

        # Debug HV details
        ref_point = _reference_point(problem_name)
        ref_hv = _reference_hv(problem_name)
        print(f"Reference point used: {ref_point}")
        print(f"Reference HV: {ref_hv}")

        # Check if solutions are dominated by ref point
        dominated = np.all(res.F <= ref_point, axis=1)
        print(f"Number of solutions valid (<= ref_point): {np.sum(dominated)}")
    else:
        print("No solutions found (res.F is None)")


if __name__ == "__main__":
    run_repro()

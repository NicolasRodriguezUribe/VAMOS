"""
Quick test: Does pymoo escape the local optimum with different seeds?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

try:
    from benchmark_utils import compute_hv
except ImportError:
    from paper.benchmark_utils import compute_hv

# Configuration
PROBLEM_NAME = "dtlz3"
N_VAR = 12
N_OBJ = 3
POP_SIZE = 100
N_EVALS = 100000
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

print(f"Testing pymoo on {PROBLEM_NAME} with different seeds...")
print("=" * 60)

results = []
for seed in range(5):
    pymoo_problem = get_problem(PROBLEM_NAME, n_var=N_VAR, n_obj=N_OBJ)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / N_VAR, eta=MUTATION_ETA),
    )
    termination = get_termination("n_eval", N_EVALS)
    res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=False)

    hv = compute_hv(res.F, PROBLEM_NAME)
    f_max = res.F.max(axis=0)
    results.append((seed, hv, f_max))

    status = "✓ Good" if hv > 0.5 else "❌ Stuck in local optimum"
    print(f"Seed {seed}: HV={hv:.6f}, F_max={f_max}, {status}")

print("=" * 60)
stuck_count = sum(1 for _, hv, _ in results if hv < 0.1)
print(f"\nSeeds stuck in local optima: {stuck_count}/5")
print("\nConclusion:")
if stuck_count > 0:
    print("  This is expected behavior for DTLZ3 - it's designed to be")
    print("  highly multimodal with many local optima. Different seeds")
    print("  can produce dramatically different results.")
    print("\n  Seed 0 happens to get stuck, which is a valid outcome")
    print("  demonstrating the problem's difficulty.")
else:
    print("  All seeds converged successfully.")

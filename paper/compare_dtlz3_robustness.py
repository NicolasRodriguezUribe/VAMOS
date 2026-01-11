"""
Compare VAMOS vs pymoo robustness on DTLZ3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2 as PyMooNSGAII
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from vamos import optimize, OptimizeConfig, make_problem_selection
from vamos.engine.algorithm.config import NSGAIIConfig

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

print(f"Comparing VAMOS vs pymoo robustness on {PROBLEM_NAME}")
print("=" * 70)

vamos_results = []
pymoo_results = []

for seed in range(5):
    print(f"\nSeed {seed}:")
    
    # VAMOS
    problem = make_problem_selection(PROBLEM_NAME, n_var=N_VAR, n_obj=N_OBJ).instantiate()
    algo_config = (
        NSGAIIConfig()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / N_VAR, eta=MUTATION_ETA)
        .selection("tournament")
        
        .engine("numpy")
        .fixed()
    )
    config = OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=algo_config,
        termination=("n_eval", N_EVALS),
        seed=seed,
        engine="numpy",
    )
    result = optimize(config)
    hv_vamos = compute_hv(result.F, PROBLEM_NAME)
    vamos_results.append(hv_vamos)
    print(f"  VAMOS: HV={hv_vamos:.6f}")
    
    # pymoo
    pymoo_problem = get_problem(PROBLEM_NAME, n_var=N_VAR, n_obj=N_OBJ)
    algorithm = PyMooNSGAII(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / N_VAR, eta=MUTATION_ETA),
    )
    termination = get_termination("n_eval", N_EVALS)
    res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=False)
    hv_pymoo = compute_hv(res.F, PROBLEM_NAME)
    pymoo_results.append(hv_pymoo)
    print(f"  pymoo: HV={hv_pymoo:.6f}")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

vamos_stuck = sum(1 for hv in vamos_results if hv < 0.1)
pymoo_stuck = sum(1 for hv in pymoo_results if hv < 0.1)

print(f"\nVAMOS: {vamos_stuck}/5 seeds stuck (mean HV={np.mean(vamos_results):.4f})")
print(f"pymoo: {pymoo_stuck}/5 seeds stuck (mean HV={np.mean(pymoo_results):.4f})")

print("\nPer-seed comparison:")
for i, (v, p) in enumerate(zip(vamos_results, pymoo_results)):
    diff = v - p
    winner = "VAMOS" if diff > 0.01 else ("pymoo" if diff < -0.01 else "tie")
    print(f"  Seed {i}: VAMOS={v:.4f}, pymoo={p:.4f}, diff={diff:+.4f} ({winner})")

if vamos_stuck < pymoo_stuck:
    print("\n✓ VAMOS is more robust on DTLZ3")
elif vamos_stuck > pymoo_stuck:
    print("\n✓ pymoo is more robust on DTLZ3")
else:
    print("\n≈ Both frameworks show similar robustness")

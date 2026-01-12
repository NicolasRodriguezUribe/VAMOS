"""
Debug script for dtlz3/pymoo HV=0 issue
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

# Import both implementations
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from vamos import optimize, OptimizeConfig, make_problem_selection
from vamos.engine.algorithm.config import NSGAIIConfig

# Import from current directory
try:
    from benchmark_utils import compute_hv, _reference_point, _load_reference_front
except ImportError:
    from paper.benchmark_utils import compute_hv, _reference_point, _load_reference_front

# Configuration
PROBLEM_NAME = "dtlz3"
N_VAR = 12
N_OBJ = 3
POP_SIZE = 100
N_EVALS = 100000
SEED = 0
CROSSOVER_PROB = 0.9
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

print("=" * 70)
print(f"Debugging {PROBLEM_NAME} with pymoo")
print("=" * 70)

# Run pymoo
print("\n1. Running pymoo...")
pymoo_problem = get_problem(PROBLEM_NAME, n_var=N_VAR, n_obj=N_OBJ)
algorithm = NSGA2(
    pop_size=POP_SIZE,
    crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
    mutation=PM(prob=1.0 / N_VAR, eta=MUTATION_ETA),
)
termination = get_termination("n_eval", N_EVALS)
res = minimize(pymoo_problem, algorithm, termination, seed=SEED, verbose=False)

print(f"   Solutions found: {res.X.shape[0]}")
print(f"   Objective space shape: {res.F.shape}")
print(f"   F min: {res.F.min(axis=0)}")
print(f"   F max: {res.F.max(axis=0)}")
print(f"   F mean: {res.F.mean(axis=0)}")

# Compute HV
ref_point = _reference_point(PROBLEM_NAME)
print(f"\n   Reference point: {ref_point}")

hv_pymoo = compute_hv(res.F, PROBLEM_NAME)
print(f"   Normalized HV: {hv_pymoo:.6f}")

# Check if solutions are dominated
dominated_count = 0
for i, f1 in enumerate(res.F):
    for j, f2 in enumerate(res.F):
        if i != j and np.all(f2 <= f1) and np.any(f2 < f1):
            dominated_count += 1
            break

print(f"   Dominated solutions: {dominated_count}/{len(res.F)}")

# Check constraint violations if any
if hasattr(res, "G") and res.G is not None:
    violations = np.sum(res.G > 0, axis=1)
    print(f"   Constraint violations: {np.sum(violations > 0)}/{len(res.G)}")

# Run VAMOS for comparison
print("\n2. Running VAMOS (numpy) for comparison...")
problem = make_problem_selection(PROBLEM_NAME, n_var=N_VAR, n_obj=N_OBJ).instantiate()
algo_config = (
    NSGAIIConfig()
    .pop_size(POP_SIZE)
    .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
    .mutation("pm", prob=1.0 / N_VAR, eta=MUTATION_ETA)
    .selection("tournament")
    .fixed()
)
config = OptimizeConfig(
    problem=problem,
    algorithm="nsgaii",
    algorithm_config=algo_config,
    termination=("n_eval", N_EVALS),
    seed=SEED,
    engine="numpy",
)
result = optimize(config)

print(f"   Solutions found: {result.X.shape[0]}")
print(f"   Objective space shape: {result.F.shape}")
print(f"   F min: {result.F.min(axis=0)}")
print(f"   F max: {result.F.max(axis=0)}")
print(f"   F mean: {result.F.mean(axis=0)}")

hv_vamos = compute_hv(result.F, PROBLEM_NAME)
print(f"   Normalized HV: {hv_vamos:.6f}")

# Compare
print("\n3. Comparison:")
print(f"   VAMOS HV: {hv_vamos:.6f}")
print(f"   pymoo HV: {hv_pymoo:.6f}")
print(f"   Difference: {abs(hv_vamos - hv_pymoo):.6f}")

# Check if pymoo solutions exceed reference point
exceeds_ref = np.any(res.F > ref_point, axis=1)
print(f"\n   pymoo solutions exceeding ref point: {np.sum(exceeds_ref)}/{len(res.F)}")
if np.sum(exceeds_ref) > 0:
    print(f"   Max exceedance: {(res.F[exceeds_ref] - ref_point).max(axis=0)}")

# Load reference front for visual comparison
try:
    ref_front = _load_reference_front(PROBLEM_NAME)
    print(f"\n4. Reference front stats:")
    print(f"   Points: {ref_front.shape[0]}")
    print(f"   F min: {ref_front.min(axis=0)}")
    print(f"   F max: {ref_front.max(axis=0)}")
except Exception as e:
    print(f"\n4. Could not load reference front: {e}")

print("\n" + "=" * 70)
print("Analysis complete. Key findings:")
print("=" * 70)

if hv_pymoo < 0.01:
    print("❌ pymoo HV is essentially zero!")
    print("   Possible causes:")
    print("   - All solutions exceed the reference point")
    print("   - Algorithm got stuck in local optima")
    print("   - Problem setup mismatch")
else:
    print("✓ pymoo HV looks reasonable")

print("\nSuggestion: Check if dtlz3 multimodality is causing pymoo to get stuck.")
print("DTLZ3 has 3^k-1 local Pareto fronts, making it very difficult.")

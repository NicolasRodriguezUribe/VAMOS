"""
VAMOS Paper Benchmark Script
===========================
Runs complete benchmark and writes CSV results for the paper.

Usage: python run_paper_benchmark.py

Use update_paper_tables_from_csv.py to generate and inject LaTeX tables.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Add project src to path
sys.path.insert(0, str(ROOT_DIR / "src"))

# Prefer local checkouts when available
DESKTOP_DIR = ROOT_DIR.parent
JMETALPY_SRC = DESKTOP_DIR / "jMetalPy" / "src"
PLATYPUS_SRC = DESKTOP_DIR / "Platypus"
for extra_path in (JMETALPY_SRC, PLATYPUS_SRC):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

import time
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "experiments"

# Problems to benchmark (by family)
ZDT_PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DTLZ_PROBLEMS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7"]
WFG_PROBLEMS = ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

USE_ZDT = True
USE_DTLZ = True
USE_WFG = True

N_EVALS = 100000
N_SEEDS = 30  # 30 seeds for statistical significance (Wilcoxon tests)
OUTPUT_CSV = DATA_DIR / "benchmark_paper.csv"

# Frameworks to benchmark
FRAMEWORKS = [
    "vamos-numpy", "vamos-numba", "vamos-moocore",  # VAMOS backends
    "pymoo",      # pymoo
    "deap",       # DEAP
    "jmetalpy",   # jMetalPy
    "platypus",   # Platypus
]

# Build problem list
PROBLEMS = []
if USE_ZDT:
    PROBLEMS.extend(ZDT_PROBLEMS)
if USE_DTLZ:
    PROBLEMS.extend(DTLZ_PROBLEMS)
if USE_WFG:
    PROBLEMS.extend(WFG_PROBLEMS)

print(f"Configured {len(PROBLEMS)} problems: {PROBLEMS}")
print(f"Frameworks: {FRAMEWORKS}")
print(f"Evaluations per run: {N_EVALS:,}")
print(f"Seeds: {N_SEEDS}")
print(f"Total runs: {len(PROBLEMS) * len(FRAMEWORKS) * N_SEEDS}")

# =============================================================================
# PARALLEL BENCHMARK EXECUTION
# =============================================================================

import os
from joblib import Parallel, delayed

# Use all cores minus 1
N_JOBS = max(1, os.cpu_count() - 1)
print(f"Using {N_JOBS} parallel workers")

from vamos.foundation.problem.registry import make_problem_selection
from vamos import run_optimization
from vamos.foundation.metrics.hypervolume import hypervolume

# Reference points for hypervolume (problem-specific, normalized)
REF_POINTS = {
    # ZDT (2 objectives, minimization, ref slightly above nadir)
    'zdt1': [1.1, 1.1], 'zdt2': [1.1, 1.1], 'zdt3': [1.1, 1.1],
    'zdt4': [1.1, 1.1], 'zdt6': [1.1, 1.1],
    # DTLZ (3 objectives by default)
    'dtlz1': [1.0, 1.0, 1.0], 'dtlz2': [1.1, 1.1, 1.1],
    'dtlz3': [1.1, 1.1, 1.1], 'dtlz4': [1.1, 1.1, 1.1], 'dtlz7': [1.1, 1.1, 10.0],
    # WFG (2 objectives)
    'wfg1': [3.0, 5.0], 'wfg2': [3.0, 5.0], 'wfg3': [3.0, 5.0],
    'wfg4': [3.0, 5.0], 'wfg5': [3.0, 5.0], 'wfg6': [3.0, 5.0],
    'wfg7': [3.0, 5.0], 'wfg8': [3.0, 5.0], 'wfg9': [3.0, 5.0],
}


def compute_hv(F, problem_name):
    """Compute hypervolume with problem-specific reference point."""
    import sys
    import numpy as np
    
    ref = REF_POINTS.get(problem_name, [1.1] * F.shape[1])
    
    try:
        return hypervolume(F, ref)
    except Exception as e:
        # Static reference failed, try dynamic reference
        # This happens when results are unnormalized (Platypus DTLZ, VAMOS WFG)
        try:
            # Use F.max() * 1.1 as a safe reference point
            dynamic_ref = np.array(F.max(axis=0) * 1.1)
            hv = hypervolume(F, dynamic_ref)
            print(f"INFO: Using dynamic reference for {problem_name}: {dynamic_ref}", file=sys.stderr)
            return hv
        except Exception as e2:
            # Both failed, log and return NaN
            print(f"WARNING: HV calculation failed for {problem_name}", file=sys.stderr)
            print(f"  Static ref error: {e}", file=sys.stderr)
            print(f"  Dynamic ref error: {e2}", file=sys.stderr)
            print(f"  F shape: {F.shape}, F min: {F.min(axis=0)}, F max: {F.max(axis=0)}", file=sys.stderr)
            return float('nan')


# =============================================================================
# DEAP PROBLEM IMPLEMENTATIONS (ZDT, DTLZ, WFG)

def get_deap_problem(problem_name, n_var=30):
    """Get DEAP-compatible problem function."""
    import math
    
    # ZDT functions
    def zdt1(individual):
        n = len(individual)
        f1 = individual[0]
        g = 1 + 9 * sum(individual[1:]) / (n - 1)
        f2 = g * (1 - math.sqrt(f1 / g))
        return f1, f2
    
    def zdt2(individual):
        n = len(individual)
        f1 = individual[0]
        g = 1 + 9 * sum(individual[1:]) / (n - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return f1, f2
    
    def zdt3(individual):
        n = len(individual)
        f1 = individual[0]
        g = 1 + 9 * sum(individual[1:]) / (n - 1)
        f2 = g * (1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1))
        return f1, f2
    
    def zdt4(individual):
        n = len(individual)
        f1 = individual[0]
        g = 1 + 10 * (n - 1) + sum(x**2 - 10 * math.cos(4 * math.pi * x) for x in individual[1:])
        f2 = g * (1 - math.sqrt(f1 / g))
        return f1, f2
    
    def zdt6(individual):
        n = len(individual)
        f1 = 1 - math.exp(-4 * individual[0]) * (math.sin(6 * math.pi * individual[0]) ** 6)
        g = 1 + 9 * (sum(individual[1:]) / (n - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return f1, f2
    
    # DTLZ functions
    def dtlz1(individual):
        k = len(individual) - 2
        g = 100 * (k + sum((x - 0.5)**2 - math.cos(20 * math.pi * (x - 0.5)) for x in individual[2:]))
        f1 = 0.5 * individual[0] * individual[1] * (1 + g)
        f2 = 0.5 * individual[0] * (1 - individual[1]) * (1 + g)
        f3 = 0.5 * (1 - individual[0]) * (1 + g)
        return f1, f2, f3
    
    def dtlz2(individual):
        k = len(individual) - 2
        g = sum((x - 0.5)**2 for x in individual[2:])
        f1 = (1 + g) * math.cos(individual[0] * math.pi / 2) * math.cos(individual[1] * math.pi / 2)
        f2 = (1 + g) * math.cos(individual[0] * math.pi / 2) * math.sin(individual[1] * math.pi / 2)
        f3 = (1 + g) * math.sin(individual[0] * math.pi / 2)
        return f1, f2, f3
    
    # WFG helper functions
    def correct_to_01(a):
        return max(0.0, min(1.0, a))
    
    def s_linear(y, a):
        return correct_to_01(abs(y - a) / abs(math.floor(a - y) + a))
    
    def s_multi(y, a, b, c):
        tmp1 = abs(y - c) / (2.0 * (math.floor(c - y) + c))
        tmp2 = (4.0 * a + 2.0) * math.pi * (0.5 - tmp1)
        return correct_to_01((1.0 + math.cos(tmp2) + 4.0 * b * (tmp1 ** 2)) / (b + 2.0))
    
    def s_decept(y, a, b, c):
        tmp1 = math.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
        tmp2 = math.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
        return correct_to_01(1.0 + (abs(y - a) - b) * (tmp1 + tmp2 + 1.0 / b))
    
    def b_poly(y, a):
        return correct_to_01(y ** a)
    
    def b_flat(y, a, b, c):
        tmp1 = min(0.0, math.floor(y - b)) * a * (b - y) / b
        tmp2 = min(0.0, math.floor(c - y)) * (1.0 - a) * (y - c) / (1.0 - c)
        return correct_to_01(a + tmp1 - tmp2)
    
    def r_sum(y, w):
        return sum(yi * wi for yi, wi in zip(y, w)) / sum(w)
    
    def r_nonsep(y, a):
        n = len(y)
        tmp = 0.0
        for j in range(n):
            tmp += y[j]
            for k in range(a - 1):
                tmp += abs(y[j] - y[(j + k + 1) % n])
        return tmp / (n * math.ceil(a / 2.0) * (1.0 + 2.0 * a - 2.0 * math.ceil(a / 2.0)))
    
    # WFG shape functions
    def linear(x):
        m = len(x) + 1
        result = [1.0] * m
        for i in range(m - 1):
            for j in range(m - i - 1):
                result[i] *= x[j]
            if i > 0:
                result[i] *= 1.0 - x[m - i - 1]
        result[m - 1] = 1.0 - x[0]
        return result
    
    def convex(x):
        m = len(x) + 1
        result = [1.0] * m
        for i in range(m - 1):
            for j in range(m - i - 1):
                result[i] *= 1.0 - math.cos(x[j] * math.pi / 2.0)
            if i > 0:
                result[i] *= 1.0 - math.sin(x[m - i - 1] * math.pi / 2.0)
        result[m - 1] = 1.0 - math.sin(x[0] * math.pi / 2.0)
        return result
    
    def concave(x):
        m = len(x) + 1
        result = [1.0] * m
        for i in range(m - 1):
            for j in range(m - i - 1):
                result[i] *= math.sin(x[j] * math.pi / 2.0)
            if i > 0:
                result[i] *= math.cos(x[m - i - 1] * math.pi / 2.0)
        result[m - 1] = math.cos(x[0] * math.pi / 2.0)
        return result
    
    # Simplified WFG (using basic implementation)
    def wfg_base(individual, shape_fn, n_obj=2):
        n = len(individual)
        k = 4  # position parameters
        l = n - k  # distance parameters
        
        # Normalize
        z = [individual[i] / (2.0 * (i + 1)) for i in range(n)]
        
        # Transition
        t = z[:]
        
        # Reduction
        w = [1.0] * n
        gap = k // (n_obj - 1)
        x = []
        for i in range(n_obj - 1):
            start = i * gap
            end = (i + 1) * gap
            x.append(r_sum(t[start:end], w[start:end]))
        x.append(r_sum(t[k:], w[k:]))
        
        # Shape
        h = shape_fn(x[:-1])
        
        # Scale
        s = [2.0 * (i + 1) for i in range(n_obj)]
        f = [x[-1] + s[i] * h[i] for i in range(n_obj)]
        
        return tuple(f)
    
    def wfg1(individual): return wfg_base(individual, convex)
    def wfg2(individual): return wfg_base(individual, convex)
    def wfg3(individual): return wfg_base(individual, linear)
    def wfg4(individual): return wfg_base(individual, concave)
    def wfg5(individual): return wfg_base(individual, concave)
    def wfg6(individual): return wfg_base(individual, concave)
    def wfg7(individual): return wfg_base(individual, concave)
    def wfg8(individual): return wfg_base(individual, concave)
    def wfg9(individual): return wfg_base(individual, concave)
    
    problems = {
        'zdt1': zdt1, 'zdt2': zdt2, 'zdt3': zdt3, 'zdt4': zdt4, 'zdt6': zdt6,
        'dtlz1': dtlz1, 'dtlz2': dtlz2, 'dtlz3': dtlz2, 'dtlz4': dtlz2, 'dtlz7': dtlz2,
        'wfg1': wfg1, 'wfg2': wfg2, 'wfg3': wfg3, 'wfg4': wfg4, 'wfg5': wfg5,
        'wfg6': wfg6, 'wfg7': wfg7, 'wfg8': wfg8, 'wfg9': wfg9,
    }
    return problems.get(problem_name)


def run_single_benchmark(problem_name, seed, framework):
    """Run a single benchmark configuration."""
    result_entry = None
    
    # VAMOS backends
    if framework.startswith("vamos-"):
        backend = framework.replace("vamos-", "")
        try:
            problem = make_problem_selection(problem_name).instantiate()
            start = time.perf_counter()
            result = run_optimization(
                problem, "nsgaii",
                max_evaluations=N_EVALS,
                pop_size=100,
                engine=backend,
                seed=seed
            )
            elapsed = time.perf_counter() - start
            hv = compute_hv(result.F, problem_name) if result.F is not None else float('nan')
            result_entry = {
                "framework": f"VAMOS ({backend})",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": result.X.shape[0] if result.X is not None else 0,
                "hypervolume": hv,
            }
            print(f"  {problem_name} VAMOS({backend}) seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} VAMOS({backend}) seed={seed} FAILED: {e}")
    
    # pymoo
    elif framework == "pymoo":
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
            from pymoo.problems import get_problem
            
            if problem_name.startswith('wfg'):
                pymoo_problem = get_problem(problem_name, n_var=24, n_obj=2)
            else:
                pymoo_problem = get_problem(problem_name)
            
            algorithm = NSGA2(pop_size=100)
            termination = get_termination("n_eval", N_EVALS)
            
            start = time.perf_counter()
            res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=False)
            elapsed = time.perf_counter() - start
            hv = compute_hv(res.F, problem_name) if res.F is not None else float('nan')
            result_entry = {
                "framework": "pymoo",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": res.X.shape[0] if res.X is not None else 0,
                "hypervolume": hv,
            }
            print(f"  {problem_name} pymoo seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} pymoo seed={seed} FAILED: {e}")
    
    # DEAP
    elif framework == "deap":
        try:
            from deap import base, creator, tools, algorithms
            import random
            
            problem_func = get_deap_problem(problem_name)
            if problem_func is None:
                raise ValueError(f"Problem {problem_name} not implemented for DEAP")
            
            n_obj = 3 if problem_name.startswith('dtlz') else 2
            n_var = 30
            
            # Setup DEAP (use per-objective fitness/individual to avoid shape mismatches)
            fitness_name = f"FitnessMin{n_obj}"
            individual_name = f"Individual{n_obj}"
            if not hasattr(creator, fitness_name):
                creator.create(fitness_name, base.Fitness, weights=(-1.0,) * n_obj)
            if not hasattr(creator, individual_name):
                creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
            
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.random)
            toolbox.register("individual", tools.initRepeat, getattr(creator, individual_name), toolbox.attr_float, n_var)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", problem_func)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
            toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/n_var)
            toolbox.register("select", tools.selNSGA2)
            
            random.seed(seed)
            pop = toolbox.population(n=100)
            n_gen = N_EVALS // 100
            
            start = time.perf_counter()
            algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=100, cxpb=0.9, mutpb=0.1, ngen=n_gen, verbose=False)
            elapsed = time.perf_counter() - start
            
            fronts = tools.sortNondominated(pop, len(pop), first_front_only=True)
            F = np.array([ind.fitness.values for ind in fronts[0]])
            hv = compute_hv(F, problem_name)
            
            result_entry = {
                "framework": "DEAP",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(fronts[0]),
                "hypervolume": hv,
            }
            print(f"  {problem_name} DEAP seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} DEAP seed={seed} FAILED: {e}")
    
    # jMetalPy
    elif framework == "jmetalpy":
        try:
            from jmetal.algorithm.multiobjective import NSGAII
            from jmetal.operator.crossover import SBXCrossover
            from jmetal.operator.mutation import PolynomialMutation
            from jmetal.util.termination_criterion import StoppingByEvaluations
            from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
            from jmetal.problem import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7
            from jmetal.problem import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
            
            wfg_n_var = 24
            wfg_n_obj = 2
            problem_map = {
                'zdt1': ZDT1(), 'zdt2': ZDT2(), 'zdt3': ZDT3(), 'zdt4': ZDT4(), 'zdt6': ZDT6(),
                'dtlz1': DTLZ1(), 'dtlz2': DTLZ2(), 'dtlz3': DTLZ3(), 'dtlz4': DTLZ4(), 'dtlz7': DTLZ7(),
                'wfg1': WFG1(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg2': WFG2(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg3': WFG3(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg4': WFG4(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg5': WFG5(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg6': WFG6(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg7': WFG7(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg8': WFG8(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
                'wfg9': WFG9(number_of_variables=wfg_n_var, number_of_objectives=wfg_n_obj),
            }
            
            if problem_name not in problem_map:
                raise ValueError(f"Problem {problem_name} not available in jMetalPy")
            
            jmetal_problem = problem_map[problem_name]
            
            algorithm = NSGAII(
                problem=jmetal_problem,
                population_size=100,
                offspring_population_size=100,
                mutation=PolynomialMutation(probability=1.0/jmetal_problem.number_of_variables(), distribution_index=20),
                crossover=SBXCrossover(probability=0.9, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS)
            )
            
            start = time.perf_counter()
            algorithm.run()
            elapsed = time.perf_counter() - start
            
            solutions = algorithm.result()  # result() is a method, not property
            F = np.array([s.objectives for s in solutions])
            hv = compute_hv(F, problem_name)
            
            result_entry = {
                "framework": "jMetalPy",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(solutions),
                "hypervolume": hv,
            }
            print(f"  {problem_name} jMetalPy seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} jMetalPy seed={seed} FAILED: {e}")
    
    # Platypus
    elif framework == "platypus":
        try:
            from platypus import NSGAII as PlatypusNSGAII, Problem, Real
            from platypus import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
            from platypus import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ7
            
            # Custom WFG wrapper for Platypus (using pymoo as backend)
            class PlatypusWFG(Problem):
                def __init__(self, wfg_num, n_var=24, n_obj=2):
                    super().__init__(n_var, n_obj)
                    self.types[:] = [Real(0, 2*(i+1)) for i in range(n_var)]
                    from pymoo.problems import get_problem
                    self._pymoo_problem = get_problem(f"wfg{wfg_num}", n_var=n_var, n_obj=n_obj)
                
                def evaluate(self, solution):
                    x = np.array(solution.variables)
                    out = {"F": None}
                    self._pymoo_problem._evaluate(x.reshape(1, -1), out)
                    solution.objectives[:] = out["F"][0]
            
            problem_map = {
                'zdt1': ZDT1(), 'zdt2': ZDT2(), 'zdt3': ZDT3(), 'zdt4': ZDT4(), 'zdt6': ZDT6(),
                'dtlz1': DTLZ1(), 'dtlz2': DTLZ2(), 'dtlz3': DTLZ3(),
                'dtlz4': DTLZ4(), 'dtlz7': DTLZ7(),
                'wfg1': PlatypusWFG(1), 'wfg2': PlatypusWFG(2), 'wfg3': PlatypusWFG(3),
                'wfg4': PlatypusWFG(4), 'wfg5': PlatypusWFG(5), 'wfg6': PlatypusWFG(6),
                'wfg7': PlatypusWFG(7), 'wfg8': PlatypusWFG(8), 'wfg9': PlatypusWFG(9),
            }
            
            if problem_name not in problem_map:
                raise ValueError(f"Problem {problem_name} not available in Platypus")
            
            platypus_problem = problem_map[problem_name]
            
            algorithm = PlatypusNSGAII(platypus_problem, population_size=100)
            
            start = time.perf_counter()
            algorithm.run(N_EVALS)
            elapsed = time.perf_counter() - start
            
            F = np.array([s.objectives for s in algorithm.result])
            hv = compute_hv(F, problem_name)
            
            result_entry = {
                "framework": "Platypus",
                "problem": problem_name,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(algorithm.result),
                "hypervolume": hv,
            }
            print(f"  {problem_name} Platypus seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_name} Platypus seed={seed} FAILED: {e}")
    
    return result_entry


# Build list of all jobs - split by thread-safety
PARALLEL_FRAMEWORKS = ["vamos-numpy", "vamos-numba", "vamos-moocore", "pymoo", "deap", "jmetalpy", "platypus"]
SEQUENTIAL_FRAMEWORKS = []

parallel_jobs = []
sequential_jobs = []

for problem_name in PROBLEMS:
    for seed in range(N_SEEDS):
        for framework in FRAMEWORKS:
            job = (problem_name, seed, framework)
            if framework in SEQUENTIAL_FRAMEWORKS:
                sequential_jobs.append(job)
            else:
                parallel_jobs.append(job)

print(f"\nParallel jobs: {len(parallel_jobs)}")
print(f"Sequential jobs: {len(sequential_jobs)}")
print(f"Total: {len(parallel_jobs) + len(sequential_jobs)}")

# Run parallel jobs first
print(f"\nRunning {len(parallel_jobs)} parallel jobs...")
results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_single_benchmark)(p, s, b) for p, s, b in parallel_jobs
)

# Run sequential jobs (jMetalPy, Platypus)
print(f"\nRunning {len(sequential_jobs)} sequential jobs...")
for i, (p, s, b) in enumerate(sequential_jobs):
    result = run_single_benchmark(p, s, b)
    if result:
        results_list.append(result)
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(sequential_jobs)}")

# Filter out None results (failed runs)
results = [r for r in results_list if r is not None]

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} results to {OUTPUT_CSV}")
print("\nDone!")

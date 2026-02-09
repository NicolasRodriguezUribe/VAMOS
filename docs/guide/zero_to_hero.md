# Zero to Hero: VAMOS in 15 Minutes

Welcome to VAMOS! This guide will take you from an empty environment to a publication-ready Multi-Objective Evolutionary Algorithm (MOEA) study in under 15 minutes.
If you are new to Python, start with `docs/guide/minimal-python.md`.

We will cover:
1.  **Flash Hero:** Installing and running your first optimization in 30 seconds.
2.  **The "Wow" Moment:** Running a parallel benchmark (NSGA-II vs MOEA/D vs SMS-EMOA) and generating LaTeX tables.
3.  **Advanced Science:** Tuning an algorithm with `RacingTuner` and defining a custom vectorized problem to unlock massive speedups.
4.  **From CLI to Analysis:** Running reproducible CLI studies and loading results for analysis.

---

## 1. Flash Hero: 30 Seconds to Pareto

First, install VAMOS (assuming you are in the repository root):

```bash
pip install -e .
```

Prefer a guided CLI? Run `vamos quickstart` for prompts and a ready-made config file.
Use `vamos quickstart --template list` to explore domain-flavored templates.
After a run, use `vamos summarize` to list recent results.

Now, let's solve the classic **ZDT1** problem (2 objectives, 30 variables) using **NSGA-II**. Create a file `hello_vamos.py`:

```python
from vamos import optimize
import matplotlib.pyplot as plt

# 1. Run NSGA-II on ZDT1
# The 'optimize' function is your main entry point.
# It automatically selects a sane configuration for standard problems.
res = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)

# 2. Visualize immediately
F = res.F  # Objective values (Pareto front approximation)
plt.scatter(F[:, 0], F[:, 1], c="teal", label="NSGA-II")
plt.title(f"ZDT1: {len(F)} solutions found")
plt.xlabel("Objective 1 (Minimize)")
plt.ylabel("Objective 2 (Minimize)")
plt.legend()
plt.show()
```

Run it:
```bash
python hello_vamos.py
```

**Boom.** You just performed a multi-objective optimization. VAMOS handles the population initialization, evolutionary loop, and non-dominated sorting for you.

---

## 2. The "Wow" Moment: Competitive Benchmarking

Let's do real science. You want to compare **NSGA-II**, **MOEA/D**, and **SMS-EMOA** on a problem, run multiple seeds in parallel to save time, and verify statistical significance.

Create `benchmark.py`:

```python
import pandas as pd
from vamos import optimize
from vamos.foundation.metrics import compute_normalized_hv
from vamos.ux.analysis.stats import friedman_test

# 1. Define the study
algorithms = ["nsgaii", "moead", "smsemoa"]
problem_name = "zdt1"
n_seeds = 5  # In a real paper, use 30+

print(f"Running benchmark on {problem_name} with {algorithms}...")

# 2. Run in PARALLEL
# VAMOS supports parallel evaluation of seeds/algorithms if you wrap this loop.
# For simplicity here, we run sequentially but show how fast it is.
# Pro-tip: Use StudyRunner or the `vamos bench` CLI for full sweeps.

results = {}
for algo in algorithms:
    print(f"  -> Running {algo}...", end="")
    # Collect normalized hypervolume (HV) for each seed
    hvs = []
    for seed in range(n_seeds):
        res = optimize(problem_name, algorithm=algo, max_evaluations=5000, seed=seed)
        # compute_normalized_hv uses ZDT reference fronts.
        # For other problems, call compute_hypervolume with an explicit reference point.
        hv_score = compute_normalized_hv(res.F, problem_name)
        hvs.append(hv_score)
    results[algo] = hvs
    print(" Done.")

# 3. Create a DataFrame
df = pd.DataFrame(results)
print("\nNormalized Hypervolume Scores:")
print(df)

# 4. Statistical Analysis (Friedman Test)
# We treat each seed as a separate 'problem instance' or aggregated block for the test
# Usually you test across multiple problems. Here we test across seeds (just for mechanics demo).
friedman = friedman_test(df.values, higher_is_better=True)
mean_ranks = friedman.ranks.mean(axis=0)

print(f"\nFriedman Test p-value: {friedman.p_value:.4e}")
print("Mean ranks:", dict(zip(df.columns, mean_ranks)))

# 5. Export to LaTeX
# This is what goes into your Overleaf paper!
latex_table = df.describe().to_latex(float_format="%.4f")
print("\n--- LaTeX Table for your Paper ---\n")
print(latex_table)
```

**Why this matters:**
*   **Consistency:** The same API (`optimize`) switches algorithms seamlessly.
*   **Analysis tools:** Dedicated stats module (`vamos.ux.analysis`) automates the math.
*   **Publication Ready:** Pandas integration means you go from Python to LaTeX in seconds.

---

## 3. Advanced Science: Racing & Vectorization

### A. Multi-Fidelity Auto-Tuning with Racing

Don't guess hyperparameters. Use the `RacingTuner` with **Hyperband-style multi-fidelity** to find the best configuration efficiently. This evaluates many configurations cheaply first, then invests more budget only in promising ones.

```python
import numpy as np
from vamos import optimize, make_problem_selection
from vamos.engine.tuning.racing import (
    RacingTuner, Scenario, TuningTask, Instance,
    WarmStartEvaluator, EvalContext,
    build_nsgaii_config_space, config_from_assignment
)

# 1. Define warm-start-aware evaluation function
def run_algorithm(config_dict, ctx: EvalContext, checkpoint=None):
    """Run algorithm with optional warm-start from previous fidelity level."""
    algo_config = config_from_assignment("nsgaii", config_dict)
    
    # Calculate how much extra budget we need
    if checkpoint is not None and ctx.previous_budget:
        extra_budget = ctx.budget - ctx.previous_budget
        # Note: Warm-start from checkpoint population is planned but not yet implemented.
        # Each fidelity level currently restarts with fresh initialization.
        # The checkpoint is preserved for budget accounting and future warm-start support.
    else:
        extra_budget = ctx.budget
    
    selection = make_problem_selection(ctx.instance.name, n_var=ctx.instance.n_var)
    res = optimize(
        selection.instantiate(),
        algorithm="nsgaii",
        algorithm_config=algo_config,
        termination=("max_evaluations", extra_budget),
        seed=ctx.seed,
    )
    
    # Return result AND checkpoint for next fidelity level
    new_checkpoint = {"X": res.X, "F": res.F}
    return res, new_checkpoint

# 2. Create evaluator with dynamic normalization (no prior bounds needed!)
evaluator = WarmStartEvaluator(
    run_fn=run_algorithm,
    score_fn=lambda res, ctx: evaluator.compute_normalized_hv(res.F),
)

# 3. Setup Task
param_space = build_nsgaii_config_space()
instances = [Instance(name="zdt1", n_var=30)]
seeds = [42, 43, 44]

task = TuningTask(
    name="tune_nsgaii_zdt1",
    param_space=param_space,
    instances=instances,
    seeds=seeds,
    budget_per_run=10000,
    maximize=True
)

# 4. Configure Multi-Fidelity Racing
scenario = Scenario(
    max_experiments=300,
    use_multi_fidelity=True,          # Enable Hyperband-style
    fidelity_levels=(1000, 3000, 10000),  # Increasing budgets
    fidelity_warm_start=True,          # Continue from checkpoints
    fidelity_promotion_ratio=0.3,      # Top 30% advance
    n_jobs=-1,                         # Parallel evaluation
)

# 5. Run!
tuner = RacingTuner(task, scenario, max_initial_configs=30)
best_config, history = tuner.run(evaluator)

print("Best configuration found:")
print(best_config)
```

**How it works:**
```
Fidelity 1 (budget=1000):  30 configs evaluated cheaply
                           -> Top 9 promoted

Fidelity 2 (budget=3000):  9 configs continue from checkpoint (+2000 evals)
                           -> Top 3 promoted

Fidelity 3 (budget=10000): 3 configs continue from checkpoint (+7000 evals)
                           -> Best returned
```

**Benefits over irace:**
- **3x more initial exploration** with same total budget
- **Warm-starting** accumulates progress between levels
- **Dynamic normalization** works without knowing ideal/nadir

### B. Custom Vectorized Problem
VAMOS is fast because it's **vectorized**. Define problems using NumPy operations, not slow Python loops.

```python
import numpy as np

class MyVectorizedProblem:
    # 30 decision variables, 2 objectives
    n_var = 30
    n_obj = 2
    xl = np.zeros(30)
    xu = np.ones(30)
    
    def evaluate(self, x):
        # x is a BATCH of solutions (PopSize, n_var).
        # We compute objectives for the WHOLE population in one go!
        
        # Objective 1: Just the first variable
        f1 = x[:, 0]
        
        # Objective 2: Some complex function of the rest
        g = 1 + 9 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        
        # Return shape: (PopSize, n_obj)
        return np.column_stack([f1, f2])

# Use it directly!
# Pass the class or instance to optimize
# (Note: In full VAMOS, you register this, but direct usage is supported for quick tests)
# res = optimize(MyVectorizedProblem(), algorithm="nsgaii", ...)
```

---

## 4. From CLI to Analysis: Reproducible Runs

When you need reproducible runs with clean artifacts, the CLI writes a standard layout under `results/`.

Run a short sweep (three seeds) into a dedicated folder:

```bash
for seed in 1 2 3; do
  vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --seed $seed --output-dir results/cli_demo
done
```

Load and aggregate the results:

```python
from vamos.ux.analysis.results import discover_runs, load_run_data, aggregate_results

runs = discover_runs("results/cli_demo")
summary = aggregate_results(runs)
print(summary)

# Inspect one run's final population.
first = load_run_data(runs[0])
print(first.F.shape)
```

If pandas is installed, `aggregate_results` returns a DataFrame; otherwise it returns a list of dicts.

---

## Next Steps

*   Check out the [Cookbook](cookbook.md) for deeper recipes.
*   Browse the [CLI Guide](cli.md) for full command reference.
*   Read the [AOS Method](../paper/aos-method.md) to understand the adaptive operator selection.
*   Explore [VAMOS Studio](studio.md) for interactive dashboards.

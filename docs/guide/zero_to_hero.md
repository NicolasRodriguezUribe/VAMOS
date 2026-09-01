# Zero to Hero: VAMOS in 15 Minutes

Welcome to VAMOS! This guide will take you from an empty environment to a publication-ready Multi-Objective Evolutionary Algorithm (MOEA) study in under 15 minutes.
If you are new to Python, start with `docs/guide/minimal-python.md`.

We will cover:
1.  **Flash Hero:** Installing and running your first optimization in 30 seconds.
2.  **The "Wow" Moment:** Running a durable benchmark (NSGA-II vs MOEA/D vs SMS-EMOA) and generating LaTeX tables.
3.  **Advanced Science:** Tuning an algorithm with `RacingTuner` and defining a custom vectorized problem to unlock massive speedups.
4.  **From CLI to Analysis:** Running reproducible CLI studies and loading results for analysis.

---

## 1. Flash Hero: 30 Seconds to Pareto

First, install VAMOS (assuming you are in the repository root):

```bash
pip install -e ".[analysis]"
```

Notes:
- Plotting in this guide requires the `analysis` extra (matplotlib/pandas).
- For accelerated kernels and distributed evaluation: `pip install -e ".[compute]"`.
- For model-based tuning backends (optuna/smac3/bohb): `pip install -e ".[tuning]"`.

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

Let's do real science. You want to compare **NSGA-II**, **MOEA/D**, and **SMS-EMOA** on a problem, preserve multiple seeds durably, and verify statistical significance.

Create `benchmark.py`:

```python
import pandas as pd
from vamos import StudySpec, create_study, load_result
from vamos.foundation.quality_indicators import compute_normalized_hv
from vamos.ux.api import friedman_test

# 1. Define the study
algorithms = ["nsgaii", "moead", "smsemoa"]
problem_name = "zdt1"
n_seeds = 5  # In a real paper, use 30+

print(f"Running benchmark on {problem_name} with {algorithms}...")

# 2. Create and execute one durable sequential study.
spec = StudySpec(
    problems=[problem_name],
    algorithms=algorithms,
    seeds=list(range(n_seeds)),
    max_evaluations=5000,
    on_error="continue",
)
completed = create_study(spec, output="studies/zdt1-comparison").run()
inspection = completed.inspect()
summary = completed.summarize()
print(inspection.state, inspection.counts)

# 3. Derive the scientific table by following StudySummary run references.
records = []
for row in summary.rows:
    run = load_result((completed.root / row.run_manifest_path).parent)
    records.append(
        {
            "algorithm": row.algorithm_id,
            "seed": row.seed,
            "hypervolume": compute_normalized_hv(run.F, problem_name),
            "study_id": row.study_id,
            "plan_id": row.plan_id,
            "task_id": row.task_id,
            "run_id": row.selected_run_id,
        }
    )

# 4. Create a DataFrame. IDs keep every derived value attributable.
df = pd.DataFrame(records).pivot(index="seed", columns="algorithm", values="hypervolume")
print("\nNormalized Hypervolume Scores:")
print(df)

# 5. Statistical Analysis (Friedman Test)
# We treat each seed as a separate 'problem instance' or aggregated block for the test
# Usually you test across multiple problems. Here we test across seeds (just for mechanics demo).
friedman = friedman_test(df.values, higher_is_better=True)
mean_ranks = friedman.ranks.mean(axis=0)

print(f"\nFriedman Test p-value: {friedman.p_value:.4e}")
print("Mean ranks:", dict(zip(df.columns, mean_ranks)))

# 6. Export to LaTeX
# This is what goes into your Overleaf paper!
latex_table = df.describe().to_latex(float_format="%.4f")
print("\n--- LaTeX Table for your Paper ---\n")
print(latex_table)
```

**Why this matters:**
*   **Consistency:** One immutable `StudySpec` freezes the complete comparison.
*   **Traceability:** Every derived row retains study, plan, task, and run IDs.
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
        max_evaluations=extra_budget,
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
    fidelity_warm_start=True,          # Pass checkpoints between fidelity levels (eval_fn opt-in)
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

Fidelity 2 (budget=3000):  9 configs re-evaluated at higher budget (+2000 evals)
                           (optionally warm-started from checkpoints)
                           -> Top 3 promoted

Fidelity 3 (budget=10000): 3 configs re-evaluated at full budget (+7000 evals)
                           (optionally warm-started from checkpoints)
                           -> Best returned
```

**Benefits over irace:**
- **3x more initial exploration** with same total budget
- **Checkpoint hooks** for warm-starting between fidelity levels (when your eval_fn consumes/returns checkpoints)
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

Status note: as of March 31, 2026, the standard run-oriented CLI path is smoke-tested again for NSGA-II/ZDT1. The commands below reflect the intended artifact layout and are reasonable onboarding examples for the common path.

Run a short sweep (three seeds) into a dedicated folder:

```bash
for seed in 1 2 3; do
  vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --seed $seed --output-root results/cli_demo
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
*   Explore [VAMOS Studio](studio.md) for interactive dashboards.

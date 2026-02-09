# Hyperparameter Tuning

VAMOS provides powerful tools for tuning algorithm hyperparameters, from simple random search to advanced racing methods.

## Programmatic Tuning

For full control, you can use the `vamos.engine.tuning` module directly in your scripts.

### Basic Random Search

```python
from vamos.engine.tuning import (
    EvalContext,
    Instance,
    ParamSpace,
    RandomSearchTuner,
    Real,
    TuningTask,
)

# 1. Define the parameter space
space = ParamSpace(
    params={
        "crossover_prob": Real("crossover_prob", 0.5, 1.0),
        "mutation_eta": Real("mutation_eta", 5.0, 50.0),
    }
)

# 2. Define the tuning task (objective function)
def evaluate_config(config: dict[str, float], ctx: EvalContext) -> float:
    # Run your algorithm with 'config' at ctx.budget and ctx.seed
    # Return a scalar quality metric (e.g., hypervolume)
    return -hypervolume  # minimize negative HV

task = TuningTask(
    name="random_search_demo",
    param_space=space,
    instances=[Instance(name="zdt1", n_var=30)],
    seeds=[0, 1],
    budget_per_run=1000,
    maximize=False,
)

# 3. Run the tuner
tuner = RandomSearchTuner(task=task, max_trials=50, seed=0)
best_config, history = tuner.run(evaluate_config)
```

### Racing Tuner

The `RacingTuner` is more efficient for stochastic algorithms. It evaluates configurations on multiple problem instances (or seeds) and discards poor performers early (statistical racing).

```python
from vamos.engine.tuning import RacingTuner, Scenario

tuner = RacingTuner(
    task,
    scenario=Scenario(
        max_experiments=50,  # Total tuning budget (config evals)
        min_survivors=3,
        n_jobs=4,            # Parallel execution
    ),
)
best_config, history = tuner.run(evaluate_config)
```

## Command Line Interface (`vamos tune`)

You can also run tuning jobs directly from the command line.

```bash
vamos tune --algorithm nsgaii --problem zdt1 --budget 1000 --n-jobs 4
```

This will run a racing tuner to find the best hyperparameters for NSGA-II on ZDT1.

Notes:
- The MOEA/D tuner explores both SBX and DE crossovers, and can select PBI aggregation (with a tunable theta). These settings align with the jMetalPy default configuration when chosen.
- The NSGA-II tuner includes external archive controls. When `archive_unbounded=True`, the archive is unbounded, `archive_type`/`archive_size_factor` are inactive, and the archive size is treated as the initial capacity (defaults to `pop_size`).

### Options

- `--algorithm`: Algorithm to tune (e.g., nsgaii, moead).
- `--problem`: Problem to tune on (e.g., zdt1, dtlz2).
- `--budget`: Total tuning budget (evaluations).
- `--n-jobs`: Number of parallel workers.
- `--output`: Directory to save the best configuration (JSON).

## Ablation Planning

Use the ablation planning helpers to define variants (same algorithm, different configuration toggles)
and build a reproducible task matrix.

```python
from vamos.engine.tuning import AblationVariant, build_ablation_plan

variants = [
    AblationVariant(name="baseline"),
    AblationVariant(name="aos"),
    AblationVariant(name="tuned", config_overrides={"population_size": 80}),
]

plan = build_ablation_plan(
    problems=["zdt1", "dtlz2"],
    variants=variants,
    seeds=[1, 2, 3],
    default_max_evals=20000,
    engine="numpy",
)
```

To convert the plan into runnable study tasks (or run directly), use the study helpers:

```python
from vamos.experiment.study.api import run_ablation_plan

results, variant_names = run_ablation_plan(
    plan,
    algorithm="nsgaii",
    base_config={"population_size": 50},
)
```

To execute an ablation plan with the experiment layer, see:
- `examples/tuning/ablation_runner.py`
- `notebooks/2_advanced/32_ablation_planning.ipynb`

Interpreting contributions: compare median final metrics (e.g., HV at full budget)
and compute deltas vs the baseline variant.

### Ablation config schema (CLI)

Use `vamos ablation --config <path>` with a YAML/JSON config. Required fields:
- `problems`: list of problem keys
- `variants`: list of variant blocks (each has a `name`)
- `seeds`: list of integer seeds
- `default_max_evals`: per-run evaluation budget

Optional fields:
- `algorithm` (default: nsgaii)
- `engine`
- `base_config` (merged into every variant before running)
- `output_root` (base output root; per-variant subfolders are created by default)
- `per_variant_output_root` (default: true)
- `output_root_by_variant` (map of variant name -> output root override)
- `budget_by_problem`, `budget_by_variant`, `budget_overrides`
- `nsgaii_variation`, `moead_variation`, `smsemoa_variation` per variant (algorithm-specific)
- `summary_dir` or `summary_path` (CSV output; default: `<output_root>/summary/ablation_metrics.csv`)

Example:

```yaml
algorithm: nsgaii
engine: numpy
output_root: results/ablation_demo
default_max_evals: 2000
problems: [zdt1]
seeds: [1, 2]
base_config:
  population_size: 60
variants:
  - name: baseline
  - name: aos
    nsgaii_variation:
      adaptive_operator_selection:
        enabled: true
summary_dir: results/ablation_demo/summary
```

Algorithm-specific variations live inside each variant block:

```yaml
variants:
  - name: moead_pbi
    moead_variation:
      aggregation:
        method: pbi
        theta: 5.0
  - name: smsemoa_fast
    smsemoa_variation:
      mutation:
        method: pm
        prob: "1/n"
```

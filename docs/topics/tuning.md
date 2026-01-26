# Hyperparameter Tuning

VAMOS provides powerful tools for tuning algorithm hyperparameters, from simple random search to advanced racing methods.

## Programmatic Tuning

For full control, you can use the `vamos.engine.tuning.api` module directly in your scripts.

### Basic Random Search

```python
from vamos.engine.tuning.api import ParamSpace, RandomSearchTuner, TuningTask

# 1. Define the parameter space
space = ParamSpace()
space.add_float("crossover_prob", 0.5, 1.0)
space.add_float("mutation_eta", 5.0, 50.0)

# 2. Define the tuning task (objective function)
def evaluate_config(config):
    # Run your algorithm with 'config'
    # Return a scalar quality metric (e.g., hypervolume)
    return -hypervolume  # minimize negative HV

task = TuningTask(
    evaluator=evaluate_config,
    space=space,
    budget=50
)

# 3. Run the tuner
tuner = RandomSearchTuner(task)
best_config = tuner.tune()
```

### Racing Tuner

The `RacingTuner` is more efficient for stochastic algorithms. It evaluates configurations on multiple problem instances (or seeds) and discards poor performers early (statistical racing).

```python
from vamos.engine.tuning.api import RacingTuner

tuner = RacingTuner(
    task,
    n_initial=10,       # Initial pool of configurations
    n_survivors=3,      # Number of configs to keep for final rounds
    n_jobs=4            # Parallel execution
)
best_config = tuner.tune()
```

## Command Line Interface (`vamos-tune`)

You can also run tuning jobs directly from the command line.

```bash
vamos-tune --algorithm nsgaii --problem zdt1 --budget 1000 --n-jobs 4
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
from vamos.engine.tuning.api import AblationVariant, build_ablation_plan

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

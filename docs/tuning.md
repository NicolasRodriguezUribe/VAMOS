# Hyperparameter Tuning

VAMOS provides powerful tools for tuning algorithm hyperparameters, from simple random search to advanced racing methods.

## Programmatic Tuning

For full control, you can use the `vamos.tuning` module directly in your scripts.

### Basic Random Search

```python
from vamos.tuning import ParamSpace, RandomSearchTuner, TuningTask

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
from vamos.tuning import RacingTuner

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

### Options

- `--algorithm`: Algorithm to tune (e.g., nsgaii, moead).
- `--problem`: Problem to tune on (e.g., zdt1, dtlz2).
- `--budget`: Total tuning budget (evaluations).
- `--n-jobs`: Number of parallel workers.
- `--output`: Directory to save the best configuration (JSON).

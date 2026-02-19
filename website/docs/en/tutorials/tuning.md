# Hyperparameter Tuning Tutorial

Use Optuna to automatically find better algorithm configurations for your problem.

---

## Why tune?

Default parameters work across many problems but are not optimal for any specific one. Tuning `pop_size`, `crossover_eta`, and `mutation_eta` for your problem can significantly improve result quality within the same evaluation budget.

---

## Install Optuna

```bash
pip install "vamos-optimization[optuna]"
# or: pip install optuna
```

---

## Basic tuning loop

Define an Optuna objective that calls `optimize()` and returns a quality metric:

```python
import optuna
import numpy as np
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)

def optuna_objective(trial):
    pop_size      = trial.suggest_int("pop_size", 50, 300, step=10)
    crossover_eta = trial.suggest_float("crossover_eta", 5.0, 40.0)
    mutation_eta  = trial.suggest_float("mutation_eta", 5.0, 40.0)

    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=5000,
        seed=42,
        algorithm_kwargs={
            "pop_size": pop_size,
            "crossover_eta": crossover_eta,
            "mutation_eta": mutation_eta,
        },
    )

    # Minimize IGD proxy: average min-distance to a reference set
    ref = np.column_stack([np.linspace(0, 1, 100), 1 - np.sqrt(np.linspace(0, 1, 100))])
    dists = np.min(np.linalg.norm(result.F[:, None] - ref[None, :], axis=2), axis=0)
    return float(dists.mean())

study = optuna.create_study(direction="minimize")
study.optimize(optuna_objective, n_trials=50, show_progress_bar=True)

print("Best parameters:", study.best_params)
print("Best IGD proxy:", study.best_value)
```

---

## Using the best parameters

Apply the tuned configuration to a full run:

```python
best = study.best_params

result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=10000,
    seed=0,
    algorithm_kwargs=best,
)

print(result.F.shape)
```

---

## Tuning multiple seeds for robustness

Average over several seeds to avoid overfitting to a single random outcome:

```python
def robust_objective(trial):
    pop_size      = trial.suggest_int("pop_size", 50, 300, step=10)
    crossover_eta = trial.suggest_float("crossover_eta", 5.0, 40.0)
    mutation_eta  = trial.suggest_float("mutation_eta", 5.0, 40.0)

    scores = []
    for seed in [0, 1, 2, 3, 4]:
        result = optimize(
            problem,
            algorithm="nsgaii",
            max_evaluations=5000,
            seed=seed,
            algorithm_kwargs={
                "pop_size": pop_size,
                "crossover_eta": crossover_eta,
                "mutation_eta": mutation_eta,
            },
        )
        scores.append(result.F[:, 0].min() + result.F[:, 1].min())  # simple proxy

    return float(np.mean(scores))

study = optuna.create_study(direction="minimize")
study.optimize(robust_objective, n_trials=30)
print(study.best_params)
```

---

## Choosing a sampler

Optuna's default sampler (TPE) works well. For a quick baseline, use random search:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.RandomSampler(seed=0),
)
study.optimize(optuna_objective, n_trials=100)
```

For expensive problems where each trial is costly, CMA-ES often finds good configurations faster:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.CmaEsSampler(seed=0),
)
study.optimize(optuna_objective, n_trials=50)
```

---

## Tuning a different algorithm

The same pattern applies to any VAMOS algorithm. Example for MOEA/D:

```python
def moead_objective(trial):
    n_neighbors          = trial.suggest_int("n_neighbors", 5, 40)
    prob_neighbor_mating = trial.suggest_float("prob_neighbor_mating", 0.5, 1.0)

    result = optimize(
        "zdt1",
        algorithm="moead",
        max_evaluations=10000,
        seed=42,
        algorithm_kwargs={
            "n_neighbors": n_neighbors,
            "prob_neighbor_mating": prob_neighbor_mating,
        },
    )

    ref = np.column_stack([np.linspace(0, 1, 100), 1 - np.sqrt(np.linspace(0, 1, 100))])
    dists = np.min(np.linalg.norm(result.F[:, None] - ref[None, :], axis=2), axis=0)
    return float(dists.mean())
```

---

## Recommended workflow

1. **Baseline**: run with defaults, record quality metric.
2. **Quick search**: TPE sampler, 50 trials, single seed — identify promising parameter regions.
3. **Robust search**: average over 5 seeds, 100 trials — reduce noise.
4. **Validate**: re-run best config on 10+ independent seeds, compare to baseline.

---

## Next steps

<div class="grid cards" markdown>

-   :material-book-open-variant: **Algorithm Reference**

    ---

    Understand each algorithm's parameter space before tuning.

    [:octicons-arrow-right-24: Algorithms](../algorithms/index.md)

-   :material-api: **API Reference**

    ---

    Full `optimize()` and `make_problem()` signatures.

    [:octicons-arrow-right-24: API Reference](../api/index.md)

</div>

# Experimental hyperparameter tuning

Tuning and statistical-analysis APIs remain experimental in VAMOS 1.0.0. The
stable optimization entry point accepts public typed algorithm configurations,
which can also be used from an external tuner.

## Install

```bash
python -m pip install "vamos-optimization[tuning]"
```

## Small Optuna loop

```python
import numpy as np
import optuna

from vamos import make_problem, optimize
from vamos.algorithms import NSGAIIConfig

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)

reference = np.column_stack(
    [np.linspace(0, 1, 100), 1 - np.sqrt(np.linspace(0, 1, 100))]
)


def objective(trial: optuna.Trial) -> float:
    pop_size = trial.suggest_int("pop_size", 20, 60, step=20)
    crossover_eta = trial.suggest_float("crossover_eta", 10.0, 30.0)
    mutation_eta = trial.suggest_float("mutation_eta", 10.0, 30.0)

    config = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .selection("tournament")
        .crossover("sbx", prob=1.0, eta=crossover_eta)
        .mutation("pm", prob=0.5, eta=mutation_eta)
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=400,
        seed=42,
    )
    distances = np.min(
        np.linalg.norm(result.F[:, None] - reference[None, :], axis=2),
        axis=0,
    )
    return float(distances.mean())


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
print(study.best_params)
```

Five trials keep the example light; scientific tuning needs an independently
designed train/validation protocol, multiple seeds, explicit indicator
semantics, and enough budget for the problem.

## Apply reviewed parameters

Construct the same public typed configuration with the selected values, then
run independent validation seeds. Do not treat the tuning trials themselves as
an unbiased performance estimate.

VAMOS also exposes experimental `vamos tune` backends. Their availability can
be checked with:

```bash
vamos tune --list-backends
```

Those experimental commands and their output are outside the stable 1.x CLI
contract.

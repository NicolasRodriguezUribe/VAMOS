# Core Module

This directory contains the foundational infrastructure for VAMOS.

## Key Components

| File | Purpose |
|------|---------|
| `runner.py` | `run_single()` — main experiment execution |
| `optimize.py` | `optimize()` — programmatic API |
| `experiment_config.py` | `ExperimentConfig` dataclass |
| `execution.py` | Algorithm execution loop |
| `hv_stop.py` | Hypervolume early stopping |
| `metadata.py` | Run metadata generation |
| `io_utils.py` | File I/O helpers |

## run_single() Flow

```
run_single(engine, algorithm, selection, config, ...)
    │
    ├── resolve_kernel(engine)           # Get backend
    ├── build_algorithm(algorithm, ...)  # Create algorithm instance
    ├── selection.instantiate()          # Create problem
    │
    └── algorithm.run(problem, termination, seed)
            │
            ├── initialize_population()
            └── while not terminated:
                    ├── ask()   → X_off
                    ├── evaluate(X_off)
                    └── tell(F_off)
```

## ExperimentConfig

```python
from vamos.foundation.core.runner import ExperimentConfig

cfg = ExperimentConfig(
    population_size=100,
    offspring_population_size=100,  # optional, defaults to pop_size
    max_evaluations=25000,
    seed=42,
)
```

## optimize() API

Higher-level API with config objects:
```python
from vamos.foundation.core.optimize import optimize, OptimizeConfig

result = optimize(OptimizeConfig(
    problem=problem,
    algorithm="nsgaii",
    algorithm_config=cfg,
    termination=("n_eval", 10000),
    seed=42,
))
```

## Output Structure

Results written to `results/{problem}/{algorithm}_seed{N}/`:
- `front.csv` — Pareto front objectives
- `variables.csv` — Decision variables (optional)
- `metadata.json` — Run configuration and timing

# Core Module

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


This directory contains the foundational infrastructure for VAMOS.

## Key Components

| File | Purpose |
|------|---------|
| `../experiment/runner.py` | `run_single()` - main experiment execution |
| `../experiment/optimize.py` | `optimize()` - programmatic API |
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
from vamos.foundation.core.experiment_config import ExperimentConfig

cfg = ExperimentConfig(
    population_size=100,
    offspring_population_size=100,  # optional, defaults to pop_size
    max_evaluations=25000,
    seed=42,
)
```

## optimize()

Higher-level API with config objects:
```python
from vamos import optimize

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=cfg,
    termination=("max_evaluations", 10000),
    seed=42,
)
```

## Output Structure

Results written to `results/{problem}/{algorithm}_seed{N}/`:
- `front.csv` — Pareto front objectives
- `variables.csv` — Decision variables (optional)
- `metadata.json` — Run configuration and timing

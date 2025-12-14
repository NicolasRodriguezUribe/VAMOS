# Tuning Module

This directory contains meta-optimization for algorithm hyperparameter tuning.

## Architecture

The tuning system uses an **outer NSGA-II** to search algorithm hyperparameter space:

```
┌─────────────────────────────────────────────────────────┐
│ Meta-level NSGA-II (searches config space)              │
│   ↓ config                                              │
│   ┌─────────────────────────────────────────────────┐   │
│   │ Inner optimization (runs algorithm on problems) │   │
│   │   → returns indicator (HV, IGD, etc.)           │   │
│   └─────────────────────────────────────────────────┘   │
│   ↑ fitness                                             │
└─────────────────────────────────────────────────────────┘
```

## Core Components

| Module | Purpose |
|--------|---------|
| `core/parameter_space.py` | `AlgorithmConfigSpace` — define tunable params |
| `evolver/tuner.py` | `NSGAIITuner` — outer optimization loop |
| `racing/` | Sequential racing for early termination |

## AlgorithmConfigSpace

```python
from vamos.engine.tuning import AlgorithmConfigSpace

space = AlgorithmConfigSpace()
space.add_int("pop_size", 50, 200)
space.add_float("crossover_prob", 0.7, 1.0)
space.add_categorical("crossover", ["sbx", "blx_alpha"])

# Or use a template
space = AlgorithmConfigSpace.from_template("nsgaii", "default")
```

## NSGAIITuner

```python
from vamos.engine.tuning import NSGAIITuner

tuner = NSGAIITuner(
    config_space=space,
    problems=[problem1, problem2],
    ref_fronts=[None, None],
    indicators=["hv"],
    max_evals_per_problem=10000,
    n_runs_per_problem=3,
    meta_population_size=20,
    meta_max_evals=100,
    seed=42,
)

X_meta, F_meta, configs, diagnostics = tuner.optimize()
```

## Config Templates

Pre-defined search spaces in `core/templates/`:
- `nsgaii/default.yaml` — standard NSGA-II parameters
- `moead/default.yaml` — MOEA/D specific

## Key Files

| File | Purpose |
|------|---------|
| `core/parameter_space.py` | AlgorithmConfigSpace class |
| `core/templates/` | Pre-defined config space YAML |
| `evolver/tuner.py` | NSGAIITuner implementation |
| `evolver/meta_problem.py` | Wraps inner optimization as problem |
| `racing/sequential.py` | Early stopping via racing |

## Adding New Tunable Parameters

1. Add to `AlgorithmConfigSpace` definition
2. Wire to algorithm config builder in `vamos.engine.algorithm.config`
3. Update templates if applicable
4. Test with small budget on ZDT1

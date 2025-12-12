# Copilot Instructions for VAMOS

> Vectorized Architecture for Multiobjective Optimization Studies - Python 3.10+

## Architecture Overview

VAMOS is a research-grade multi-objective optimization framework with **vectorized kernels**. Core data flow:

1. **Problem** (`vamos.problem.registry`) → defines bounds, objectives, constraints
2. **Algorithm** (`vamos.algorithm/`) → NSGAII, MOEAD, SMSEMOA, SPEA2, IBEA, SMPSO, NSGA3
3. **Kernel** (`vamos.kernel/`) → NumPy, Numba, or MooCore backends for fast operations
4. **Runner** (`vamos.core.runner`) → orchestrates experiments, writes to `results/`

Algorithms use **array-based populations** (X=decision vars, F=objectives, G=constraints) — never per-individual object instances.

## Critical Patterns

### Problem Creation
```python
from vamos.problem.registry import make_problem_selection
selection = make_problem_selection("zdt1", n_var=30)
problem = selection.instantiate()
```

### Algorithm Configuration (Builder Pattern)
```python
from vamos.algorithm.config import NSGAIIConfig
cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .engine("numpy")
    .fixed())
```

### Running Experiments
```python
from vamos.core.runner import run_single, ExperimentConfig
from vamos.core.optimize import optimize, OptimizeConfig
```

### External Archives
Algorithms support bounded archives for preserving elite solutions:
```python
run_single(
    engine_name="moocore",
    algorithm_name="nsgaii",
    selection=selection,
    config=exp_cfg,
    external_archive_size=100,
    archive_type="hypervolume",  # or "crowding"
)
```
- `HypervolumeArchive`: SMS-EMOA style pruning by HV contribution
- `CrowdingDistanceArchive`: jMetal style pruning by crowding distance

### Tuning / AutoNSGA-II
Meta-optimization over algorithm hyperparameters:
```python
from vamos.tuning import AlgorithmConfigSpace, NSGAIITuner

space = AlgorithmConfigSpace()
space.add_float("crossover_prob", 0.7, 1.0)
space.add_categorical("crossover", ["sbx", "blx_alpha"])
tuner = NSGAIITuner(space, problem, budget=10000)
best_config = tuner.run()
```
See [examples/auto_nsga2_tuning_example.py](../examples/auto_nsga2_tuning_example.py) for full usage.

## Key Conventions

- **Vectorization required**: Hot paths must use NumPy/kernel ops, not Python loops
- **Workspace pattern**: Operators use `VariationWorkspace` for efficient memory reuse
- **Seeded RNG**: Pass `np.random.default_rng(seed)` through call stacks, avoid global RNG
- **Explicit imports**: No wildcard imports; structure as stdlib → third-party → local
- **Type hints everywhere**: All public APIs must have full type annotations

## Developer Commands

```powershell
# Setup
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[backends,benchmarks,dev]"

# Verify installation
python -m vamos.diagnostics.self_check

# Run tests (skip slow)
pytest -m "not slow"

# Quick smoke test
python -m vamos.cli.main --problem zdt1 --max-evaluations 2000

# Format/lint before commit
black src tests; ruff check src tests
```

## File Organization

| Path | Purpose |
|------|---------|
| [src/vamos/algorithm/](../src/vamos/algorithm/) | Algorithm cores + config builders (nsgaii.py, config.py) |
| [src/vamos/kernel/](../src/vamos/kernel/) | Backend kernels (numpy_backend.py, numba_backend.py) |
| [src/vamos/operators/](../src/vamos/operators/) | Variation operators by encoding (real/, binary.py, permutation.py) |
| [src/vamos/problem/registry/](../src/vamos/problem/registry/) | Problem specs + `make_problem_selection()` |
| [src/vamos/core/](../src/vamos/core/) | Runner, optimize API, metadata, I/O utilities |
| [src/vamos/tuning/](../src/vamos/tuning/) | Meta-optimizer, config spaces, AutoNSGA-II |
| [tests/](../tests/) | pytest suite mirroring src layout |

## Adding New Components

**New operator**: Copy template from [operators/real/](../src/vamos/operators/real/), implement vectorized `__call__`, register in `__init__.py`

**New problem**: Define in [problem/](../src/vamos/problem/), implement `evaluate(X) → F` or `evaluate_population`, register in `PROBLEM_SPECS`

**New algorithm config**: Add dataclass in [algorithm/config.py](../src/vamos/algorithm/config.py), create builder class with fluent API

## Testing Guidelines

- Unit tests: deterministic components (operators, kernels, registries)
- Smoke tests: tiny budgets (pop=10, evals=100), assert finite non-empty fronts
- Stochastic: use statistical/property checks, not exact value matching

## Avoid

- Adding heavy dependencies without justification (see `pyproject.toml` extras)
- Removing vectorization or replacing kernels with pure Python loops
- Hardcoded paths or storing large artifacts under `src/`
- Breaking existing public APIs without updating all call sites

## Config Files (YAML/JSON)

Run experiments via `--config path.yaml`:
```yaml
defaults:
  algorithm: nsgaii
  engine: numpy
  population_size: 100
  max_evaluations: 20000

problems:
  zdt1:
    n_var: 30
  dtlz2:
    algorithm: moead
    n_obj: 3

nsgaii:
  crossover: {method: sbx, prob: 0.9, eta: 20}
  mutation: {method: pm, prob: "1/n", eta: 20}

spea2:
  archive_size: 80
  crossover: {method: sbx, prob: 0.9, eta: 20}
```
CLI flags override config values. Per-problem sections override defaults.

## Related Documentation

- [AGENTS.md](../AGENTS.md) — Environment setup, repo layout, coding conventions
- [AGENTS_tasks.md](../AGENTS_tasks.md) — Task playbook for common changes (operators, problems, studies)
- [AGENTS_codex_prompts.md](../AGENTS_codex_prompts.md) — Ready-to-paste prompts for code assistants
- [CODING_GUIDELINES.md](../CODING_GUIDELINES.md) — Style, testing, and PR checklist
- [docs/](../docs/) — MkDocs site with CLI, algorithms, constraints, extending guides

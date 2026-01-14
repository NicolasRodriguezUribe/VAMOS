# Copilot Instructions for VAMOS

> Vectorized Architecture for Multiobjective Optimization Studies - Python 3.10+

## User-Friendliness First

VAMOS prioritizes **user-friendly APIs**. Always prefer the clean public interface:

```python
# ✅ PREFERRED - Clean public API
from vamos import optimize, make_problem_selection
from vamos.algorithms import NSGAIIConfig
from vamos.problems import (
    ZDT1,
    FeatureSelectionProblem,
    HyperparameterTuningProblem,
    WeldedBeamDesignProblem,
)
from vamos.ux.api import plot_pareto_front_2d, weighted_sum_scores
from vamos.engine.tuning.api import ParamSpace, RandomSearchTuner, RacingTuner

# ❌ AVOID - Internal paths (for contributors only)
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
```

## Architecture Overview

VAMOS is a research-grade multi-objective optimization framework with **vectorized kernels**. Core data flow:

1. **Problem** (`vamos.foundation.problem.registry`) → defines bounds, objectives, constraints
2. **Algorithm** (`vamos.engine.algorithm/`) → NSGAII, MOEAD, SMSEMOA, SPEA2, IBEA, SMPSO, NSGA3
3. **Kernel** (`vamos.foundation.kernel/`) → NumPy, Numba, or MooCore backends for fast operations
4. **Runner** (`vamos.foundation.core.runner`) → orchestrates experiments, writes to `results/`

Algorithms use **array-based populations** (X=decision vars, F=objectives, G=constraints) — never per-individual object instances.

## Critical Patterns

### Problem Creation
```python
# User-friendly public API
from vamos import make_problem_selection
from vamos.problems import ZDT1, FeatureSelectionProblem

# Via registry
selection = make_problem_selection("zdt1", n_var=30)
problem = selection.instantiate()

# Direct instantiation
problem = ZDT1(n_var=30)
problem = FeatureSelectionProblem(dataset="breast_cancer")
```

### Algorithm Configuration (Builder Pattern)
```python
from vamos.algorithms import NSGAIIConfig
cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", pressure=2)
    .survival("nsga2")
    .engine("numpy")
    .build())
```

### Running Experiments
```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)

cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", pressure=2)
    .survival("nsga2")
    .engine("numpy")
    .build())
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=cfg,
    termination=("n_eval", 10000),
    seed=42,
)
print(f"Found {result.F.shape[0]} solutions")
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

### Racing-Style Tuning
Hyperparameter tuning with racing (irace-inspired):
```python
from vamos.engine.tuning.api import ParamSpace, Real, Int, Categorical, RandomSearchTuner, RacingTuner

space = ParamSpace(params={
    "crossover_prob": Real(0.7, 1.0),
    "pop_size": Int(50, 200),
    "crossover": Categorical(["sbx", "blx_alpha"]),
})
# Use RandomSearchTuner or RacingTuner with TuningTask
```

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
python -m vamos.experiment.diagnostics.self_check

# Run tests (skip slow)
pytest -m "not slow"

# Quick smoke test
python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000

# Format/lint before commit
black src tests; ruff check src tests
```

## File Organization

| Path | Purpose |
|------|---------|
| [src/vamos/engine/algorithm/](../src/vamos/engine/algorithm/) | Algorithm cores + config builders (nsgaii.py, config.py) |
| [src/vamos/foundation/kernel/](../src/vamos/foundation/kernel/) | Backend kernels (numpy_backend.py, numba_backend.py) |
| [src/vamos/engine/operators/](../src/vamos/engine/operators/) | Variation operators by encoding (real/, binary.py, permutation.py) |
| [src/vamos/foundation/problem/registry/](../src/vamos/foundation/problem/registry/) | Problem specs + `make_problem_selection()` |
| [src/vamos/foundation/core/](../src/vamos/foundation/core/) | Runner, optimize API, metadata, I/O utilities |
| [src/vamos/engine/tuning/](../src/vamos/engine/tuning/) | Racing tuner, config spaces, ParamSpace |
| [tests/](../tests/) | pytest suite mirroring src layout |

## Adding New Components

**New operator**: Copy template from [engine/operators/real/](../src/vamos/engine/operators/real/), implement vectorized `__call__`, register in `__init__.py`

**New problem**: Define in [foundation/problem/](../src/vamos/foundation/problem/), implement `evaluate(X) → F` or `evaluate_population`, register in `PROBLEM_SPECS`

**New algorithm config**: Add dataclass in [engine/algorithm/config/](../src/vamos/engine/algorithm/config/), create builder class with fluent API

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

# Copilot Instructions for VAMOS

> Vectorized Architecture for Multiobjective Optimization Studies - Python 3.10+

## User-Friendliness First

VAMOS prioritizes **user-friendly APIs**. Always prefer the clean public interface:

```python
# ✅ PREFERRED - Clean public API
from vamos import (
    optimize, OptimizeConfig, NSGAIIConfig,
    ZDT1, make_problem_selection,
    FeatureSelectionProblem, HyperparameterTuningProblem,
    plot_pareto_front_2d, weighted_sum_scores,
    ParamSpace, RandomSearchTuner, RacingTuner,
)

# ❌ AVOID - Internal paths (for contributors only)
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
```

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
# User-friendly public API
from vamos import make_problem_selection, ZDT1, FeatureSelectionProblem

# Via registry
selection = make_problem_selection("zdt1", n_var=30)
problem = selection.instantiate()

# Direct instantiation
problem = ZDT1(n_var=30)
problem = FeatureSelectionProblem(dataset="breast_cancer")
```

### Algorithm Configuration (Builder Pattern)
```python
from vamos import NSGAIIConfig
cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .engine("numpy")
    .fixed())
```

### Running Experiments
```python
from vamos import optimize, OptimizeConfig, NSGAIIConfig, make_problem_selection

selection = make_problem_selection("zdt1", n_var=30)
problem = selection.instantiate()

cfg = NSGAIIConfig().pop_size(100).fixed()
result = optimize(
    OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 10000),
        seed=42,
    )
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
from vamos import ParamSpace, Real, Int, Categorical, RandomSearchTuner, RacingTuner

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

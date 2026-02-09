# Problem Module

This directory contains benchmark and real-world optimization problems for VAMOS.

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.

## Quick Custom Problems: `make_problem()`

The friendliest way to create a problem. Lives in `builder.py` and is
exported from the public API (`from vamos import make_problem`):

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], 1 - x[0] ** 0.5],
    n_var=2, n_obj=2,
    bounds=[(0, 1), (0, 1)],
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=5000)
```

`make_problem()` creates a `FunctionalProblem` that implements `ProblemProtocol`
internally. Scalar functions are auto-vectorized. Use `vectorized=True` for
batch functions. Constraints are supported via the `constraints` parameter.

CLI scaffolding: `vamos create-problem` generates a ready-to-run `.py` file.

## Problem Protocol

All problems must implement:
```python
class ProblemProtocol(Protocol):
    n_var: int
    n_obj: int
    xl: np.ndarray  # lower bounds (n_var,)
    xu: np.ndarray  # upper bounds (n_var,)
    
    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        """Evaluate population X (N × n_var), write F (N × n_obj) to out["F"]."""
        ...
```

Optional: `n_constraints`, `evaluate_constraints(X) → G`

## Adding a New Problem

**Quick / ad-hoc**: use `make_problem()` (see above) -- no registration needed.

**Reusable benchmark**: follow the registry approach:

1. Create `new_problem.py` with class implementing `ProblemProtocol`
2. Register in the appropriate family module under `registry/families/`:
   See [registry/AGENTS.md](registry/AGENTS.md) for the canonical workflow.
```python
ProblemSpec(
    key="new_problem",
    label="New Problem",
    default_n_var=30,
    default_n_obj=2,
    allow_n_obj_override=False,
    factory=lambda n_var, n_obj: NewProblem(n_var),
    encoding="continuous",  # or "binary", "permutation", "integer", "mixed"
)
```
3. Add to the `SPECS` dict in that family module (the aggregator will pick it up)
4. Add test in `tests/test_new_benchmarks.py`

## Directory Structure

| Path | Purpose |
|------|---------|
| `builder.py` | `make_problem()` convenience builder + `FunctionalProblem` wrapper |
| `types.py` | `ProblemProtocol` definition |
| `zdt*.py` | ZDT1-ZDT6 (2-obj continuous) |
| `dtlz.py` | DTLZ1-DTLZ4 (scalable n_obj) |
| `wfg.py` | WFG1-WFG9 (scalable n_obj) |
| `lz.py` | LZ09 F1-F9 |
| `cec.py`, `cec2009.py` | CEC2009 UF/CF problems |
| `binary.py` | Knapsack, feature selection, QUBO |
| `integer.py` | Job assignment, resource allocation |
| `tsp.py`, `tsplib.py` | TSP with TSPLIB parser |
| `mixed.py` | Mixed-variable design |
| `real_world/` | Engineering design, ML tuning |
| `registry/` | ProblemSpec definitions + selection API |
| `registry/families/` | Family-specific spec modules (canonical) |

## Using Problems

```python
from vamos.foundation.problem.registry import make_problem_selection

# Standard usage
selection = make_problem_selection("zdt1", n_var=30)
problem = selection.instantiate()

# Scalable objectives (DTLZ, WFG)
selection = make_problem_selection("dtlz2", n_var=12, n_obj=3)
```

## Encoding Types

| Encoding | Bounds | Example Problems |
|----------|--------|------------------|
| `continuous` | `[xl, xu]` floats | ZDT, DTLZ, WFG |
| `binary` | `{0, 1}` | Knapsack, QUBO |
| `permutation` | `[0, n-1]` unique | TSP |
| `integer` | `[xl, xu]` integers | Job assignment |
| `mixed` | combination | Mixed design |

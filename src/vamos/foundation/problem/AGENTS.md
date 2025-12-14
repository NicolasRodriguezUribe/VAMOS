# Problem Module

This directory contains benchmark and real-world optimization problems for VAMOS.

## Problem Protocol

All problems must implement:
```python
class ProblemProtocol(Protocol):
    n_var: int
    n_obj: int
    xl: np.ndarray  # lower bounds (n_var,)
    xu: np.ndarray  # upper bounds (n_var,)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate population X (N × n_var) → F (N × n_obj)"""
        ...
```

Optional: `n_constraints`, `evaluate_constraints(X) → G`

## Adding a New Problem

1. Create `new_problem.py` with class implementing `ProblemProtocol`
2. Register in [registry/specs.py](registry/specs.py):
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
3. Add to `PROBLEM_SPECS` dict in the same file
4. Add test in `tests/test_new_benchmarks.py`

## Directory Structure

| Path | Problems |
|------|----------|
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

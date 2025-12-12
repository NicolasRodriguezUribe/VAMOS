# Operators Module

This directory contains variation operators organized by encoding type.

## Structure

| Path | Encoding | Operators |
|------|----------|-----------|
| `real/` | Continuous | SBX, BLX-alpha, polynomial mutation, uniform |
| `binary.py` | Binary | Bitflip, one-point, two-point, uniform crossover |
| `permutation.py` | Permutation | PMX, OX, swap, insert, inversion |
| `integer.py` | Integer | Random reset, creep mutation |
| `mixed.py` | Mixed | Composite operators |
| `repair.py` | All | Bounds repair strategies |

## Operator Contract

All operators must be **vectorized** and follow this signature:
```python
def __call__(
    self,
    X: np.ndarray,              # (N, n_var) population
    bounds: tuple[np.ndarray, np.ndarray],  # (xl, xu)
    rng: np.random.Generator,   # seeded RNG
    workspace: VariationWorkspace | None = None,
) -> np.ndarray:                # (N, n_var) offspring
```

## VariationWorkspace Pattern

For memory efficiency, operators reuse pre-allocated arrays:
```python
from vamos.operators.real import VariationWorkspace

ws = VariationWorkspace(pop_size=100, n_var=30)
offspring = crossover(X, bounds, rng, workspace=ws)
```

## Adding a New Operator

1. Implement in appropriate file (or new file in `real/`)
2. Follow vectorized pattern — no Python loops over individuals
3. Use `rng` parameter, never global random state
4. Add to `__init__.py` exports
5. Register in `vamos.algorithm.variation` operator lookup
6. Add unit tests in `tests/operators/`

## Key Files

| File | Purpose |
|------|---------|
| `real/sbx.py` | Simulated Binary Crossover (reference impl) |
| `real/polynomial_mutation.py` | PM mutation (reference impl) |
| `real/__init__.py` | VariationWorkspace, operator exports |
| `repair.py` | Bounds handling: clip, random, bounce |

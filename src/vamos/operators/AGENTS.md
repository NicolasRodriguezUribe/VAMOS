# Operators Module

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


This directory contains operator implementations and algorithm-specific wiring.

## Structure

| Path | Encoding | Operators |
|------|----------|-----------|
| `impl/real/` | Continuous | SBX, BLX-alpha, polynomial mutation, uniform |
| `impl/binary.py` | Binary | Bitflip, one-point, two-point, uniform crossover |
| `impl/permutation.py` | Permutation | PMX, OX, swap, insert, inversion |
| `impl/integer.py` | Integer | Random reset, creep mutation |
| `impl/mixed.py` | Mixed | Composite operators |
| `impl/repair.py` | All | Bounds repair strategies |
| `policies/` | Wiring | Algorithm-specific operator selection/defaults |

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
from vamos.operators.impl.real import VariationWorkspace

ws = VariationWorkspace(pop_size=100, n_var=30)
offspring = crossover(X, bounds, rng, workspace=ws)
```

## Adding a New Operator

1. Implement in appropriate file (or new file under `impl/real/`)
2. Follow vectorized pattern — no Python loops over individuals
3. Use `rng` parameter, never global random state
4. Add to `__init__.py` exports
5. Register in `vamos.operators.impl.registry` operator lookup
6. Add unit tests in `tests/operators/`

## Key Files

| File | Purpose |
|------|---------|
| `impl/real/crossover.py` | SBX/BLX/UNDX/SPX crossovers |
| `impl/real/mutation.py` | PM/gaussian/cauchy mutations |
| `impl/real/initialize.py` | LHS/Scatter initializers |
| `impl/real/__init__.py` | VariationWorkspace, operator exports |
| `impl/repair.py` | Bounds handling: clip, random, bounce |

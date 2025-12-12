# Kernel Module

This directory contains backend kernels for compute-intensive operations.

## Architecture

Kernels abstract away performance-critical operations so algorithms remain backend-agnostic:

```python
# Algorithm code uses kernel methods
ranks, crowding = self.kernel.nsga2_ranking(F)
hv = self.kernel.hypervolume(F, ref_point)
```

## Available Backends

| Backend | Module | Requirements | Best For |
|---------|--------|--------------|----------|
| NumPy | `numpy_backend.py` | numpy (core) | Default, portable |
| Numba | `numba_backend.py` | numba | Large populations |
| MooCore | `moocore_backend.py` | moocore | Accurate HV |

## KernelBackend Protocol

All backends implement:
```python
class KernelBackend(Protocol):
    def fast_non_dominated_sort(self, F: np.ndarray) -> tuple[np.ndarray, list]:
        """Return (ranks, fronts) for objective matrix F."""
        ...
    
    def crowding_distance(self, F: np.ndarray) -> np.ndarray:
        """Compute crowding distances for a single front."""
        ...
    
    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Combined non-dominated sort + crowding."""
        ...
    
    def hypervolume(self, F: np.ndarray, ref: np.ndarray) -> float:
        """Compute hypervolume indicator."""
        ...
```

## Backend Selection

```python
from vamos.kernel.registry import resolve_kernel

kernel = resolve_kernel("numpy")  # or "numba", "moocore"
```

Via CLI: `--engine numpy` or `--engine moocore`

## Adding a New Backend

1. Create `new_backend.py` implementing `KernelBackend` protocol
2. Register in `registry.py`: `KERNELS["new"] = NewKernel`
3. Add optional dependency to `pyproject.toml` extras
4. Gate import with try/except for graceful fallback
5. Add smoke test in `tests/test_backends_smoke.py`

## Key Files

| File | Purpose |
|------|---------|
| `numpy_backend.py` | Reference NumPy implementation |
| `numba_backend.py` | JIT-compiled fast paths |
| `moocore_backend.py` | R moocore bindings |
| `registry.py` | Backend name → class mapping |
| `backend.py` | KernelBackend Protocol definition |

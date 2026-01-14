# JAX Backend

The JAX backend provides GPU/TPU acceleration for VAMOS kernels.

## Installation

```bash
# CPU-only
pip install jax

# GPU (CUDA 12)
pip install jax[cuda12]

# GPU (CUDA 11)  
pip install jax[cuda11_local]
```

Or install with VAMOS extras:
```bash
pip install -e ".[autodiff]"
```

## Usage

```python
result = vamos.optimize("zdt1", engine="jax", budget=10000)
```

Note: JAX defaults to strict ranking, which falls back to NumPy for exact Pareto fronts.
Set `VAMOS_JAX_STRICT_RANKING=0` for approximate ranking.

## When to Use JAX

| Scenario | Recommended Engine |
|----------|-------------------|
| Small populations (<500) | numpy |
| Large populations (>1000) | jax |
| GPU available | jax |
| Quick experiments | numpy |
| Production runs | jax (strict ranking) |

## Expected Speedup

| Population Size | NumPy | JAX (CPU) | JAX (GPU) |
|----------------|-------|-----------|-----------|
| 500 | 1.0x | 0.8x | 2x |
| 2000 | 1.0x | 1.5x | 10x |
| 10000 | 1.0x | 3x | 50x+ |

*Note: First run includes JIT compilation overhead.*

## Verifying GPU

```python
import jax
print(jax.devices())  # Should show GPU
```

## Limitations

- First call triggers JIT compilation (slower)
- Strict ranking falls back to NumPy for exact Pareto fronts (default; may reduce speed)
- Approximate ranking uses domination counts (set `VAMOS_JAX_STRICT_RANKING=0`)
- Complex archive operations fall back to NumPy

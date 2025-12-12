---
applyTo: "**/*.py"
description: Python coding standards for VAMOS
---

# Python Standards for VAMOS

## Style & Structure
- Follow PEP 8/PEP 257 with type hints on all public APIs
- Import order: stdlib → third-party → local (no wildcards)
- Use `np.random.default_rng(seed)` for randomness, never global RNG

## Vectorization Required
- Hot paths must use NumPy/kernel operations, not Python loops
- Operators use `VariationWorkspace` for memory-efficient batch processing
- Prefer `kernel.fast_non_dominated_sort()` over manual Pareto checks

## Error Handling
- Raise `ValueError`/`TypeError` for invalid config, not silent fallbacks
- Validate dictionary configs early with helpers like `_prepare_mutation_params()`

## Common Imports
```python
from vamos.problem.registry import make_problem_selection
from vamos.algorithm.config import NSGAIIConfig
from vamos.core.runner import run_single, ExperimentConfig
from vamos.core.optimize import optimize, OptimizeConfig
```

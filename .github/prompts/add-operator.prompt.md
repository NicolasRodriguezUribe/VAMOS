# Add New Operator

Use this prompt when adding a new variation operator (crossover or mutation) to VAMOS.

## Task
Add a new `{OPERATOR_TYPE}` operator named `{OPERATOR_NAME}` for `{ENCODING}` encoding.

## Steps

1. **Create operator file** in `src/vamos/engine/operators/{encoding}/`
   - Use existing operator as template (e.g., `sbx.py` for crossover, `polynomial_mutation.py` for mutation)
   - Implement vectorized `__call__(self, X, bounds, rng)` method
   - Return array of same shape as input

2. **Register in `__init__.py`**
   ```python
   from .{operator_name} import {OperatorClass}
   ```

3. **Wire into variation pipeline**
   - Add to `src/vamos/engine/algorithm/variation.py` operator lookup
   - Add to `src/vamos/engine/algorithm/config/variation.py` if config-driven

4. **Add tests**
   - `tests/engine/operators/{encoding}/test_{operator_name}.py`
   - Test shapes, bounds respect, determinism with seed

## Template

```python
import numpy as np
from vamos.engine.operators.real import VariationWorkspace

class {OperatorClass}:
    def __init__(self, prob: float = 0.9, **params):
        self.prob = prob
        # store other params
    
    def __call__(
        self,
        X: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
        workspace: VariationWorkspace | None = None,
    ) -> np.ndarray:
        xl, xu = bounds
        # Vectorized implementation
        # Return offspring array
        return X_off
```

## Checklist
- [ ] Vectorized implementation (no Python loops over individuals)
- [ ] Bounds handling (clip or repair)
- [ ] Uses `rng` parameter, not global random
- [ ] Type hints on all methods
- [ ] Docstring with paper reference if applicable
- [ ] Unit test with shape/bounds assertions
- [ ] Smoke test in algorithm run (optional)

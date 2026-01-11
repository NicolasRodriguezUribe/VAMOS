# Adding a new kernel backend

Kernel backends implement performance-critical primitives. Follow this template to add one.

## Required interface

Implement `KernelBackend` methods in `src/vamos/foundation/kernel/backend.py`:

```python
from __future__ import annotations
import numpy as np
from vamos.foundation.kernel.backend import KernelBackend

class MyBackend(KernelBackend):
    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def tournament_selection(self, ranks, crowding, pressure, rng, n_parents):
        ...

    def sbx_crossover(self, X_parents, params, rng, xl, xu):
        ...

    def polynomial_mutation(self, X, params, rng, xl, xu):
        ...

    def nsga2_survival(self, X, F, X_off, F_off, pop_size):
        ...
```

## Registration

Register the backend in `src/vamos/foundation/kernel/registry.py`:

```python
from importlib import import_module
from .my_backend import MyBackend

KERNELS["mybackend"] = lambda: MyBackend()
```

If it depends on optional deps, use lazy loading and clear ImportErrors (see existing registry patterns).

## Smoke test

Add a backend-marked smoke test (skip if dependency missing):

```python
import pytest
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.engine.api import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII

@pytest.mark.mybackend
def test_mybackend_smoke():
    kernel = resolve_kernel("mybackend")
    cfg = (
        NSGAIIConfig()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=1.0, eta=15)
        .mutation("pm", prob="1/n", eta=20)
        .selection("tournament", pressure=2)
        .engine("mybackend")
        .fixed()
        .to_dict()
    )
    algo = NSGAII(cfg, kernel=kernel)
    res = algo.run(ZDT1Problem(n_var=4), termination=("n_eval", 8), seed=0)
    assert res["F"].shape[0] > 0
```

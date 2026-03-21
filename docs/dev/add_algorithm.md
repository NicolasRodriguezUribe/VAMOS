# Adding a new algorithm

Use this checklist to add an algorithm that plugs into VAMOS orchestration.

## Required pieces

1) Implement the algorithm core under `src/vamos/engine/algorithm/`:

```python
from __future__ import annotations
import numpy as np

class MyAlgorithm:
    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int, eval_strategy=None, live_viz=None):
        rng = np.random.default_rng(seed)
        # ... initialize, loop, return {"X": ..., "F": ...}
        return {"X": np.empty((0, problem.n_var)), "F": np.empty((0, problem.n_obj))}
```

2) Add a config dataclass/builder under `src/vamos/engine/algorithm/config/` if needed so configs are serializable, validated, and usable from `optimize(..., algorithm_config=...)`.

3) Register the algorithm in `src/vamos/engine/algorithm/registry.py`:

```python
from .my_algorithm import MyAlgorithm
from vamos.engine.algorithm.registry import register_algorithm

register_algorithm("my_algorithm", lambda cfg, kernel: MyAlgorithm(cfg, kernel=kernel))
```

4) If you want a typed public config object or CLI-facing defaults, add those through the existing config and registry layers. Do not add a second facade.

5) Add a fast smoke test under `tests/`:

```python
def test_my_algorithm_smoke():
    from vamos.engine.algorithm.registry import resolve_algorithm
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem
    algo_ctor = resolve_algorithm("my_algorithm")
    algo = algo_ctor({"pop_size": 4}, kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=4)
    res = algo.run(problem, termination=("max_evaluations", 8), seed=1)
    assert "F" in res and res["F"].shape[1] == problem.n_obj
```

6) Document any new public knobs in `docs/topics/extending.md` or the relevant user guide page, and add a smoke test for the documented path.

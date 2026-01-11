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

2) Add a config dataclass/builder to `src/vamos/engine/algorithm/config.py` if needed so configs are serializable and validated.

3) Register the algorithm in `src/vamos/engine/algorithm/registry.py`:

```python
from .my_algorithm import MyAlgorithm
ALGORITHMS["my_algorithm"] = lambda cfg, kernel: MyAlgorithm(cfg, kernel=kernel)
```

4) (Optional) Wire it into the factory (`src/vamos/engine/algorithm/factory.py`) if you want it accessible via runner/CLI presets.

5) Add a fast smoke test under `tests/`:

```python
def test_my_algorithm_smoke():
    from vamos.engine.algorithm.registry import resolve_algorithm
    from vamos.foundation.kernel.numpy_backend import NumPyKernel
    from vamos.foundation.problem.zdt1 import ZDT1Problem
    algo_ctor = resolve_algorithm("my_algorithm")
    algo = algo_ctor({"pop_size": 4}, kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=4)
    res = algo.run(problem, termination=("n_eval", 8), seed=1)
    assert "F" in res and res["F"].shape[1] == problem.n_obj
```

6) Document any new CLI flags/config keys in `docs/algorithms.md` or relevant docs.

# Adding a new problem

This template shows the minimal pieces required to add a benchmark/problem to VAMOS.
For the canonical registry workflow, see `src/vamos/foundation/problem/registry/AGENTS.md`.

## Steps

1) Implement the problem class under `src/vamos/foundation/problem/your_family.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MyProblem:
    n_var: int = 2
    n_obj: int = 2
    xl: float | np.ndarray = 0.0
    xu: float | np.ndarray = 1.0
    encoding: str = "continuous"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        # X shape: (N, n_var)
        f1 = X[:, 0]
        f2 = 1.0 - np.sqrt(f1)
        out["F"] = np.stack([f1, f2], axis=1)
```

2) Register it in the appropriate family module under
`src/vamos/foundation/problem/registry/families/`:

```python
from ..common import ProblemSpec
from ...my_family import MyProblem

SPECS["my_problem"] = ProblemSpec(
    key="my_problem",
    label="My Problem",
    default_n_var=2,
    default_n_obj=2,
    allow_n_obj_override=False,
    description="Short description of the landscape.",
    factory=lambda n_var, _n_obj: MyProblem(n_var=n_var),
    encoding="continuous",
)
```

3) Add a minimal smoke test (fast!) under `tests/`:

```python
def test_my_problem_smoke():
    from vamos.foundation.problem.registry import make_problem_selection
    selection = make_problem_selection("my_problem", n_var=2)
    problem = selection.instantiate()
    import numpy as np
    X = np.random.rand(4, problem.n_var)
    out = {}
    problem.evaluate(X, out)
    assert "F" in out and out["F"].shape == (4, problem.n_obj)
```

4) If the problem needs reference data (fronts, weight files), add them under `src/vamos/foundation/data/` and update packaging rules.

# Adding a new problem

VAMOS offers three ways to define custom problems, from simplest to most
integrated. Pick the approach that fits your use case.

---

## Quick way: `make_problem()` (recommended for personal / ad-hoc problems)

No class, no protocol, no NumPy vectorization required. Write a plain function
that evaluates a **single** solution and VAMOS handles the rest:

```python
from vamos import make_problem, optimize

# Your function receives x (1-D array of n_var) and returns n_obj values
def my_objectives(x):
    f1 = x[0]
    f2 = (1 + x[1]) * (1 - x[0] ** 0.5)
    return [f1, f2]

problem = make_problem(
    my_objectives,
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],   # per-variable (lower, upper)
    encoding="real",
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
```

**Options:**

| Parameter | Description |
|-----------|-------------|
| `bounds` | List of `(lower, upper)` tuples, one per variable |
| `xl`, `xu` | Alternative: scalar or array bounds (mutually exclusive with `bounds`) |
| `vectorized` | Set `True` if your function already handles 2-D batches `(N, n_var) -> (N, n_obj)` |
| `encoding` | `"real"` (default), `"binary"`, `"integer"`, `"permutation"`, or `"mixed"` |
| `name` | Human-readable label (defaults to the function name) |
| `constraints` | Optional constraint function; return values where `g(x) <= 0` is feasible |
| `n_constraints` | Number of constraint values (required when `constraints` is provided) |

### Scaffold a file with the CLI wizard

```bash
vamos create-problem
```

This interactive wizard prompts for name, variables, objectives, and bounds,
then generates a ready-to-run `.py` file with `make_problem()` already wired
up. Just fill in the TODO markers and run `python my_problem.py`.

Non-interactive mode for scripting:

```bash
vamos create-problem --name "portfolio optimizer" --n-var 5 --n-obj 3 --yes
```

Use `--style class` to generate a class-based template instead.

### Visual builder in VAMOS Studio

For a fully visual experience, launch VAMOS Studio and open the
**Problem Builder** tab:

```bash
vamos studio
```

The Problem Builder lets you:

- Pick from starter templates (ZDT1-like, Schaffer, Fonseca-Fleming, etc.)
- Edit objective code in a text area with live syntax checking
- Configure algorithm, budget, population size, and bounds
- Click **Run preview** to see the Pareto front rendered instantly
- Export a standalone `.py` script when you are happy with the result

---

## Class-based: implement `ProblemProtocol` directly

For problems that need more control (custom initialization, caching, etc.),
implement the protocol directly:

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
        # X shape: (N, n_var) -- a batch of N solutions
        f1 = X[:, 0]
        f2 = 1.0 - np.sqrt(f1)
        out["F"] = np.stack([f1, f2], axis=1)
```

Pass the instance directly to `optimize()`:

```python
result = optimize(MyProblem(), algorithm="nsgaii", max_evaluations=5000)
```

---

## Registry approach: add a reusable benchmark problem

For problems that should be available by name (e.g. `optimize("my_problem")`)
and shared across the project. For the canonical registry workflow, see
`src/vamos/foundation/problem/registry/AGENTS.md`.

### Steps

1) Implement the problem class under `src/vamos/foundation/problem/your_family.py`
   (see class-based approach above).

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

4) If the problem needs reference data (fronts, weight files), add them under
   `src/vamos/foundation/data/` and update packaging rules.

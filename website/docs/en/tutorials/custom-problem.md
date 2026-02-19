# Custom Problem Tutorial

How to define, configure, and optimize custom problems in VAMOS.

---

## Minimal problem

The simplest case: a lambda with bounds.

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
print(result.F)
```

---

## Named function

Use a named function for readability or when the objective has multiple steps:

```python
import numpy as np
from vamos import make_problem, optimize

def sphere2(x):
    """Minimize distance to (0, 0) and distance to (1, 1)."""
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2
    return [f1, f2]

problem = make_problem(
    sphere2,
    n_var=2, n_obj=2,
    xl=-2.0, xu=2.0,
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=10000, seed=0)
```

`xl` and `xu` broadcast a single bound value to all variables. Use `bounds=[(xl_0, xu_0), ...]` for per-variable bounds.

---

## Per-variable bounds

```python
problem = make_problem(
    lambda x: [x[0] + x[1], x[0] ** 2 + (x[1] - 1) ** 2],
    n_var=2, n_obj=2,
    bounds=[(-5.0, 5.0), (0.0, 10.0)],   # variable 0: [-5, 5], variable 1: [0, 10]
)
```

---

## Adding constraints

Constraints follow the convention `g(x) ≤ 0` (feasible when the value is ≤ 0).

```python
from vamos import make_problem, optimize

def objectives(x):
    return [x[0] ** 2 + x[1] ** 2,
            (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2]

def constraints(x):
    # Require x[0] + x[1] >= 1.0
    # Rewrite as: 1.0 - x[0] - x[1] <= 0
    return [1.0 - x[0] - x[1]]

problem = make_problem(
    objectives,
    n_var=2, n_obj=2,
    bounds=[(-3.0, 3.0), (-3.0, 3.0)],
    constraints=constraints,
    n_constraints=1,
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=10000, seed=42)
print(result.F)
```

`n_constraints` must equal the length of the list returned by `constraints`.

---

## Multiple constraints

Return all constraint values in a single list. Each must be ≤ 0 for the solution to be feasible:

```python
def constraints(x):
    g1 = x[0] + x[1] - 2.0    # x[0] + x[1] <= 2.0
    g2 = 0.5 - x[0]           # x[0] >= 0.5
    return [g1, g2]

problem = make_problem(
    objectives,
    n_var=2, n_obj=2,
    xl=0.0, xu=3.0,
    constraints=constraints,
    n_constraints=2,
)
```

---

## Higher-dimensional problem

Scale to more variables by adjusting `n_var` and providing matching bounds:

```python
import numpy as np
from vamos import make_problem, optimize

def zdt1(x):
    f1 = x[0]
    g  = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]

problem = make_problem(zdt1, n_var=30, n_obj=2, xl=0.0, xu=1.0)
result  = optimize(problem, algorithm="nsgaii", max_evaluations=30000, seed=42)
```

---

## Using result arrays

Results are plain NumPy arrays — no custom methods required:

```python
import numpy as np

result = optimize(problem, algorithm="nsgaii", max_evaluations=10000, seed=42)

# result.F shape: (N, n_obj)
# result.X shape: (N, n_var)

# Sort by first objective
idx = np.argsort(result.F[:, 0])
print(result.F[idx[:5]])     # five solutions closest to f1=0
print(result.X[idx[0]])      # decision variables of the best f1 solution
```

---

## Next steps

<div class="grid cards" markdown>

-   :material-tune: **Hyperparameter Tuning**

    ---

    Automatically search for better algorithm parameters using Optuna.

    [:octicons-arrow-right-24: Tuning Tutorial](tuning.md)

-   :material-book-open-variant: **Algorithm Reference**

    ---

    Parameters and guidance for all nine algorithms.

    [:octicons-arrow-right-24: Algorithms](../algorithms/index.md)

</div>

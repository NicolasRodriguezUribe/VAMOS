# VAMOS Cookbook

Common recipes and patterns for using VAMOS.

## 1. Custom Problem Definition

Define a problem by inheriting from `Problem`:

```python
import numpy as np
from vamos.foundation.problem.types import ProblemProtocol

class MyProblem(ProblemProtocol):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
    
    def evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        f1 = X[:, 0]
        f2 = (1 + X[:, 1]) / X[:, 0]
        out["F"] = np.column_stack([f1, f2])

problem = MyProblem()
```

## 2. Handling Constraints

Box constraints are handled via `xl` and `xu`. For other constraints, fill `out["G"]`. VAMOS expects $g(x) \le 0$ for feasible solutions.

```python
    def evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        # ... calculate F ...
        
        # Constraint: x[0] + x[1] <= 1.5
        g1 = (X[:, 0] + X[:, 1]) - 1.5
        out["G"] = g1.reshape(-1, 1)

problem = MyProblem(n_constr=1)
```

## 3. Visualization Callback

Use `live_viz` to see progress or save frames.

```python
import matplotlib.pyplot as plt

class MyCallback:
    def __call__(self, algorithm):
        print(f"Gen {algorithm.n_gen}: {len(algorithm.pop)} solutions")
        # Access population: algorithm.pop.get("F")

optimize(config, live_viz=MyCallback())
```

## 4. Re-using Algorithm State (Checkpointing)

VAMOS algorithms are stateful. You can `resume()` them if you manually stepped them, or pickle them (ensure backends are pickleable).

*Note: Full checkpointing support is in active development.*

## 5. Using Numba for Performance

Select a backend via `optimize(..., engine=...)` or `OptimizeConfig.engine`.

```python
result = optimize(problem, algorithm="nsgaii", engine="numba")
```

## 6. Comparing Algorithms

Run multiple algorithms and plot their fronts together.

```python
import matplotlib.pyplot as plt
from vamos.api import optimize

res_nsga2 = optimize(problem, algorithm="nsgaii")
res_moead = optimize(problem, algorithm="moead")

plt.scatter(res_nsga2.F[:,0], res_nsga2.F[:,1], label="NSGA-II")
plt.scatter(res_moead.F[:,0], res_moead.F[:,1], label="MOEA/D")
plt.legend()
plt.show()
```

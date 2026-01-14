# VAMOS Cookbook

Common recipes and patterns for using VAMOS.

## Recommended path (optimize)

The `optimize(...)` API is the fastest way to run experiments in Python.

```python
from vamos import optimize
from vamos.ux.api import result_summary_text

result = optimize("zdt1", algorithm="nsgaii", budget=5000, pop_size=100, seed=0)
print(result_summary_text(result))
```

## 1. Custom Problem Definition

Define a problem by implementing `ProblemProtocol` (attributes plus `evaluate`):

```python
import numpy as np


class MyProblem:
    def __init__(self) -> None:
        self.n_var = 2
        self.n_obj = 2
        self.n_constr = 0
        self.xl = np.array([0.0, 0.0])
        self.xu = np.array([1.0, 1.0])
        self.encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        f1 = X[:, 0]
        f2 = (1.0 + X[:, 1]) / X[:, 0]
        out["F"] = np.column_stack([f1, f2])


problem = MyProblem()
```

## 2. Handling Constraints

Box constraints are handled via `xl` and `xu`. For other constraints, fill `out["G"]`. VAMOS expects g(x) <= 0 for feasible solutions. Set `n_constr` to the number of constraints if you want the count explicitly tracked.

```python
    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        # ... calculate F ...

        # Constraint: x[0] + x[1] <= 1.5
        g1 = (X[:, 0] + X[:, 1]) - 1.5
        out["G"] = g1.reshape(-1, 1)
```

## 3. Visualization Callback

Use `live_viz` to see progress or save frames.

```python
from vamos import optimize

class MyCallback:
    def __call__(self, algorithm):
        print(f"Gen {algorithm.n_gen}: {len(algorithm.pop)} solutions")
        # Access population: algorithm.pop.get("F")

optimize("zdt1", algorithm="nsgaii", budget=2000, live_viz=MyCallback())
```

## 4. Re-using Algorithm State (Checkpointing)

VAMOS algorithms are stateful. You can `resume()` them if you manually stepped them, or pickle them (ensure backends are pickleable).

*Note: Full checkpointing support is in active development.*

## 5. Using Numba for Performance

Select a backend via `optimize(..., engine=...)`.

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", engine="numba", budget=5000)
```

## 6. Comparing Algorithms

Run multiple algorithms and plot their fronts together.

```python
import matplotlib.pyplot as plt
from vamos import optimize

res_nsga2 = optimize("zdt1", algorithm="nsgaii", budget=4000)
res_moead = optimize("zdt1", algorithm="moead", budget=4000)

plt.scatter(res_nsga2.F[:, 0], res_nsga2.F[:, 1], label="NSGA-II")
plt.scatter(res_moead.F[:, 0], res_moead.F[:, 1], label="MOEA/D")
plt.legend()
plt.show()
```

## 7. Inspect Auto-Resolved Defaults

See which top-level settings were inferred vs provided.

```python
from vamos import optimize

result = optimize("zdt1")
print(result.explain_defaults())
```

## 8. Operator Facade Access

Import common operators directly from `vamos.operators`.

```python
import numpy as np
from vamos.operators import SBXCrossover, PolynomialMutation

xl = np.zeros(30)
xu = np.ones(30)
crossover = SBXCrossover(prob_crossover=0.9, eta=15.0, lower=xl, upper=xu)
mutation = PolynomialMutation(prob=1 / 30, eta=20.0, lower=xl, upper=xu)
```

## 9. Multi-Seed Studies

Pass a list of seeds to run a small study in one call.

```python
from vamos import optimize
from vamos.ux.api import result_summary_text

results = optimize("zdt1", algorithm="nsgaii", budget=4000, seed=[0, 1, 2, 3])
for idx, res in enumerate(results):
    print(idx, result_summary_text(res))
```

## 10. Algorithm Config Objects (Reproducible Runs)

Use a config object when you want every knob explicit.

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig

cfg = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .offspring_size(100)
    .crossover("sbx", prob=1.0, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", pressure=2)
    .build()
)

result = optimize("zdt1", algorithm="nsgaii", algorithm_config=cfg, budget=8000, seed=7)
```

## 11. Multiprocessing Evaluation

For expensive evaluations, use the multiprocessing backend explicitly.

```python
from vamos import optimize
from vamos.foundation.eval.backends import MultiprocessingEvalBackend

backend = MultiprocessingEvalBackend(n_workers=4)
result = optimize("zdt1", algorithm="nsgaii", budget=6000, eval_strategy=backend)
```

## 12. Hypervolume-Based Early Stopping

Stop once the hypervolume reaches a target fraction of the reference front.

```python
from vamos import optimize
from vamos.foundation.core.hv_stop import build_hv_stop_config

hv_cfg = build_hv_stop_config(hv_threshold=0.9, hv_reference_front=None, problem_key="zdt1")
hv_cfg["max_evaluations"] = 12000

result = optimize("zdt1", algorithm="nsgaii", termination=("hv", hv_cfg), seed=3)
```

## 13. Save Results for Offline Analysis

Persist results to a folder for later inspection.

```python
from vamos import optimize
from vamos.ux.api import save_result

result = optimize("zdt1", algorithm="nsgaii", budget=5000)
save_result(result, "results/zdt1_nsgaii")
```

## 14. Select a Single Solution from the Front

Pick a knee point or a simple min objective.

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", budget=5000)
choice = result.best("knee")
print(choice["F"])
```

## 15. JAX Engine (Strict Ranking Default)

Use JAX for acceleration. Strict ranking is on by default and falls back to NumPy for exact Pareto fronts.

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", engine="jax", budget=5000)
```

Set `VAMOS_JAX_STRICT_RANKING=0` for approximate ranking if you want maximum speed.

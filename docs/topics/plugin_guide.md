# VAMOS Plugin Guide

This guide explains how to extend VAMOS with custom algorithms, operators, and problems.

## Custom Algorithm

Implement the `AlgorithmLike` protocol and register it:

```python
from vamos.engine.algorithm.registry import ALGORITHMS, AlgorithmLike
from vamos.foundation.kernel.backend import KernelBackend

class MyAlgorithm:
    def __init__(self, config: dict, kernel: KernelBackend):
        self.config = config
        self.kernel = kernel

    def run(self, problem, termination, seed, eval_strategy=None, live_viz=None):
        # Your optimization loop here
        return {"F": final_objectives, "X": final_solutions, "evaluations": n_eval}

# Register with the global registry
@ALGORITHMS.register("my_algorithm")
def build_my_algorithm(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return MyAlgorithm(cfg, kernel)
```

Then use it:
```python
from vamos import optimize
result = optimize("zdt1", algorithm="my_algorithm", budget=1_000, pop_size=100, seed=42)
```

## Custom Operator

Register with `operator_registry`:

```python
from vamos.operators.impl.registry import operator_registry
import numpy as np

class MyMutation:
    def __init__(self, prob: float = 0.1, **kwargs):
        self.prob = prob

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        # Your mutation logic
        return X

operator_registry.register("my_mutation", MyMutation)
```

## Custom Problem

Implement `ProblemProtocol`:

```python
from vamos.foundation.problem.types import ProblemProtocol
import numpy as np

class MyProblem(ProblemProtocol):
    n_var = 10
    n_obj = 2
    n_constr = 0
    xl = np.zeros(10)
    xu = np.ones(10)
    encoding = "continuous"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        f1 = X[:, 0]
        f2 = 1 - np.sqrt(f1)
        out["F"][:, 0] = f1
        out["F"][:, 1] = f2
```

Use it directly:
```python
from vamos import optimize
result = optimize(MyProblem(), algorithm="nsgaii", budget=1_000, pop_size=100, seed=42)
```

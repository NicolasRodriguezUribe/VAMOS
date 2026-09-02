# Quickstart tutorial

This tutorial uses the stable VAMOS 1.0.0 Python facades.

## 1. Run and inspect a benchmark

```python
import numpy as np

from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)

order = np.argsort(result.F[:, 0])
sorted_f = result.F[order]
print(sorted_f.shape)
print(sorted_f[0])
```

## 2. Define a scalar custom problem

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)

custom_result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)
```

The default adapter evaluates the scalar callable one solution at a time. It
does not claim vectorized batch performance.

## 3. Compare algorithms with fixed inputs

```python
from vamos import optimize

for algorithm in ("nsgaii", "spea2", "smpso", "ibea"):
    candidate = optimize(
        "zdt1",
        algorithm=algorithm,
        max_evaluations=400,
        pop_size=40,
        seed=42,
    )
    print(algorithm, candidate.F.shape)
```

A shared seed and budget improve comparability but do not by themselves prove
that different algorithms are scientifically equivalent. Record all resolved
configuration and use multiple independent seeds for research claims.

## 4. Many objectives

```python
result = optimize(
    "dtlz2",
    algorithm="nsgaiii",
    n_obj=5,
    max_evaluations=420,
    seed=42,
)

print(result.F.shape[1])  # 5 objectives
```

Reference-direction algorithms may derive a population size from objective
count. VAMOS rejects incompatible explicit sizes instead of silently changing
them.

## Next steps

- [Custom problems](custom-problem.md)
- [Experimental tuning](tuning.md)
- [API reference](../api/index.md)

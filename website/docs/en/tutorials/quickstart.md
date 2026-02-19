# Quickstart Tutorial

A complete walk-through of common VAMOS workflows from install to result analysis.

---

## Prerequisites

- Python 3.9+
- `pip install vamos-optimization`
- Basic NumPy familiarity

---

## What is a Pareto front?

Single-objective optimization finds one best value. Multi-objective optimization has two or more conflicting objectives — improving one typically worsens another. The result is a *Pareto front*: the set of solutions where no objective can be improved without degrading another. VAMOS returns the entire front. You choose the trade-off.

---

## Step 1 — Run a benchmark

ZDT1 is a standard 2-objective test function with 30 variables:

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)

print(f"Solutions found: {result.F.shape[0]}")
print(f"Objective range f1: [{result.F[:, 0].min():.4f}, {result.F[:, 0].max():.4f}]")
print(f"Objective range f2: [{result.F[:, 1].min():.4f}, {result.F[:, 1].max():.4f}]")
```

---

## Step 2 — Inspect the result

`result.F` and `result.X` are NumPy arrays:

```python
import numpy as np

# Sort by objective 1
order = np.argsort(result.F[:, 0])
sorted_F = result.F[order]
sorted_X = result.X[order]

# Extreme points
print("Minimum f1:", sorted_F[0])    # lowest f1, highest f2
print("Minimum f2:", sorted_F[-1])   # highest f1, lowest f2
```

---

## Step 3 — Define a custom problem

Wrap any Python function with `make_problem()`:

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
print(result.F)
```

The function takes a 1-D array and returns a list of floats. VAMOS auto-vectorizes it over the population internally.

---

## Step 4 — Try a different algorithm

Change `algorithm=` to switch:

```python
for algo in ["nsgaii", "spea2", "smpso", "ibea"]:
    r = optimize("zdt1", algorithm=algo, max_evaluations=10000, seed=42)
    print(f"{algo:10s}: {r.F.shape[0]} solutions, f1_min={r.F[:, 0].min():.4f}")
```

---

## Step 5 — Control algorithm parameters

Pass keyword arguments to configure the algorithm:

```python
result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=10000,
    seed=42,
    algorithm_kwargs={
        "pop_size": 200,
        "crossover_eta": 30,
        "mutation_eta": 25,
    },
)
```

---

## Step 6 — Many objectives with NSGA-III

Switch to a many-objective algorithm for 4+ objectives:

```python
result = optimize(
    "dtlz2",
    algorithm="nsgaiii",
    max_evaluations=50000,
    seed=42,
    algorithm_kwargs={"n_partitions": 8},
)

print(result.F.shape)   # (45, 3) — depends on reference point count
```

---

## Next steps

<div class="grid cards" markdown>

-   :material-tools: **Custom Problem**

    ---

    Constraints, integer variables, multi-dimensional bounds.

    [:octicons-arrow-right-24: Custom Problem Tutorial](custom-problem.md)

-   :material-tune: **Hyperparameter Tuning**

    ---

    Automatically find better algorithm configurations using Optuna.

    [:octicons-arrow-right-24: Tuning Tutorial](tuning.md)

-   :material-api: **API Reference**

    ---

    Full signatures for `optimize()`, `make_problem()`, and the result object.

    [:octicons-arrow-right-24: API Reference](../api/index.md)

</div>

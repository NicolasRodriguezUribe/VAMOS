# Getting Started

Install VAMOS, run your first optimization, and understand the result — in under five minutes.

---

## Install

Install the core package from PyPI:

```bash
pip install vamos-optimization
```

Optional extras:

```bash
pip install "vamos-optimization[numba]"      # Numba JIT-compiled operators
pip install "vamos-optimization[pandas]"     # DataFrame export
pip install "vamos-optimization[matplotlib]" # Pareto front plotting
pip install "vamos-optimization[studio]"     # VAMOS Studio (Streamlit dashboard)
pip install "vamos-optimization[optuna]"     # Hyperparameter tuning
pip install "vamos-optimization[all]"        # All of the above
```

**Requirements:** Python 3.9+ · NumPy ≥ 1.23 · SciPy ≥ 1.9 · joblib ≥ 1.2

---

## Your first optimization

The entry point is `optimize()`. Pass a problem name and get a result back:

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)
```

`"zdt1"` is a built-in benchmark. `max_evaluations=10000` is the evaluation budget. `seed=42` makes the run reproducible.

---

## Reading the result

`result` holds the non-dominated solutions found:

```python
print(result.F)          # objective values,   shape (n_solutions, 2)
print(result.X)          # decision variables, shape (n_solutions, 30)
print(result.F.shape)    # (100, 2) for ZDT1 with pop_size=100
```

`result.F` and `result.X` are plain NumPy arrays. All standard NumPy operations apply:

```python
import numpy as np

best_f1 = result.F[np.argmin(result.F[:, 0])]   # solution with lowest f1
spread   = result.F[:, 0].max() - result.F[:, 0].min()
```

---

## Defining a custom problem

Use `make_problem()` to wrap any Python function:

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
print(result.F)
```

The function takes a 1-D array `x` and returns a list of objective values. VAMOS handles vectorization internally — no NumPy broadcasting needed.

---

## Algorithm selection

| Keyword | Algorithm | Best for |
|---------|-----------|----------|
| `"nsgaii"` | NSGA-II | 2–3 objectives, general-purpose |
| `"nsgaiii"` | NSGA-III | 4+ objectives |
| `"moead"` | MOEA/D | Decomposition, scalable subproblems |
| `"smsemoa"` | SMS-EMOA | High hypervolume quality |
| `"spea2"` | SPEA2 | Archive-based diversity |
| `"ibea"` | IBEA | Indicator-based selection |
| `"smpso"` | SMPSO | Particle swarm, smooth fronts |
| `"agemoea"` | AGE-MOEA | Adaptive geometry, unknown front shape |
| `"rvea"` | RVEA | Reference vector, many-objective |

Change algorithm with one keyword:

```python
result = optimize("dtlz2", algorithm="nsgaiii", max_evaluations=30000, seed=0)
```

---

## Algorithm-specific parameters

Pass extra parameters via `algorithm_kwargs`:

```python
result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=10000,
    seed=42,
    algorithm_kwargs={"pop_size": 200, "crossover_eta": 30},
)
```

See each [algorithm's page](algorithms/index.md) for the full parameter list.

---

## Parallel evaluation

For expensive objective functions, distribute evaluation across CPU cores using joblib:

```python
result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=10000,
    eval_strategy="multiprocessing",
    seed=42,
)
```

`eval_strategy="serial"` (default) runs on one core. `"multiprocessing"` uses joblib to parallelize the evaluation of individuals within each generation.

---

## Next steps

<div class="grid cards" markdown>

-   :material-school: **Tutorials**

    ---

    Step-by-step guides for constraints, integer variables, and hyperparameter tuning.

    [:octicons-arrow-right-24: Tutorials](tutorials/quickstart.md)

-   :material-book-open-variant: **Algorithm Reference**

    ---

    Parameters, characteristics, and citations for all nine algorithms.

    [:octicons-arrow-right-24: Algorithms](algorithms/index.md)

-   :material-api: **API Reference**

    ---

    Full signatures for `optimize()`, `make_problem()`, and the result object.

    [:octicons-arrow-right-24: API](api/index.md)

</div>

# API Reference

Complete reference for VAMOS public functions. Import from the top-level `vamos` package.

---

## `optimize()`

Main entry point for running an optimization.

```python
from vamos import optimize

result = optimize(
    problem,
    *,
    algorithm="nsgaii",
    max_evaluations=None,
    seed=42,
    algorithm_kwargs=None,
    eval_strategy=None,
    engine=None,
    verbose=False,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `problem` | `str \| problem` | required | Benchmark name (e.g. `"zdt1"`) or a problem object from `make_problem()`. |
| `algorithm` | `str` | `"nsgaii"` | Algorithm keyword. One of: `"nsgaii"`, `"nsgaiii"`, `"moead"`, `"smsemoa"`, `"spea2"`, `"ibea"`, `"smpso"`, `"agemoea"`, `"rvea"`. |
| `max_evaluations` | `int \| None` | `None` | Total objective function evaluation budget. If `None`, a default is chosen based on `n_var`. |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `algorithm_kwargs` | `dict \| None` | `None` | Algorithm-specific parameters (e.g. `{"pop_size": 200, "crossover_eta": 30}`). |
| `eval_strategy` | `str \| None` | `None` | Evaluation backend: `"serial"` (default) or `"multiprocessing"` (joblib-based, multicore). |
| `engine` | `str \| None` | `None` | Computation backend: `"numpy"` (default) or `"numba"` (requires `pip install numba`). |
| `verbose` | `bool` | `False` | Print progress to stdout every generation. |

### Returns

`OptimizationResult` — the non-dominated solutions and metadata.

### Examples

```python
# One-liner
result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)

# Custom problem
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)

# Explicit algorithm parameters
result = optimize(
    "dtlz2",
    algorithm="nsgaiii",
    max_evaluations=50000,
    seed=0,
    algorithm_kwargs={"pop_size": 200, "crossover_eta": 30},
)
```

---

## `make_problem()`

Wrap a Python function as a VAMOS problem.

```python
from vamos import make_problem

problem = make_problem(
    fn,
    *,
    n_var,
    n_obj,
    bounds=None,
    xl=None,
    xu=None,
    constraints=None,
    n_constraints=0,
    name=None,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `fn` | `Callable` | required | Objective function. Takes a 1-D array `x` of length `n_var`, returns a list of `n_obj` values. |
| `n_var` | `int` | required | Number of decision variables. |
| `n_obj` | `int` | required | Number of objectives (all minimized). |
| `bounds` | `list[tuple[float, float]] \| None` | `None` | Per-variable bounds: `[(xl_0, xu_0), ..., (xl_n, xu_n)]`. Takes precedence over `xl`/`xu`. |
| `xl` | `float \| array \| None` | `None` | Lower bound(s). Scalar broadcasts to all variables. |
| `xu` | `float \| array \| None` | `None` | Upper bound(s). Scalar broadcasts to all variables. |
| `constraints` | `Callable \| None` | `None` | Constraint function: `g(x) → list`. Values ≤ 0 are feasible. |
| `n_constraints` | `int` | `0` | Number of constraint values returned by `constraints`. |
| `name` | `str \| None` | `None` | Human-readable problem name (used in logs and result metadata). |

### Examples

```python
# Minimal — lambda with bounds
problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)

# Named function
def sphere2(x):
    return [x[0] ** 2 + x[1] ** 2,
            (x[0] - 1) ** 2 + (x[1] - 1) ** 2]

problem = make_problem(sphere2, n_var=2, n_obj=2, xl=-2.0, xu=2.0)

# With inequality constraints  (g(x) <= 0 is feasible)
def objectives(x):
    return [x[0] ** 2, (x[0] - 1) ** 2 + x[1] ** 2]

def constraints(x):
    return [x[0] + x[1] - 1.0]   # x[0] + x[1] <= 1.0

problem = make_problem(
    objectives,
    n_var=2, n_obj=2,
    bounds=[(0, 1), (0, 1)],
    constraints=constraints,
    n_constraints=1,
)
```

---

## `OptimizationResult`

Returned by `optimize()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `F` | `ndarray` | Objective values. Shape `(N, n_obj)`. |
| `X` | `ndarray` | Decision variables. Shape `(N, n_var)`. |
| `meta` | `dict` | Metadata: algorithm, seed, config, evaluation counts, IGD (if benchmark). |

### Usage

```python
result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)

result.F          # shape (100, 2)
result.X          # shape (100, 30)
result.meta       # {"algorithm": "nsgaii", "seed": 42, "n_evals": 10000, ...}

# Standard NumPy on the arrays
import numpy as np
best_f1_idx = np.argmin(result.F[:, 0])
print(result.F[best_f1_idx])    # solution with lowest f1
print(result.X[best_f1_idx])    # corresponding decision variables
```

---

## Built-in benchmark problems

Pass any of these strings as the `problem` argument to `optimize()`:

**ZDT suite (2 objectives)**

| Name | Variables | Description |
|------|:---------:|-------------|
| `"zdt1"` | 30 | Convex Pareto front |
| `"zdt2"` | 30 | Concave Pareto front |
| `"zdt3"` | 30 | Disconnected Pareto front |
| `"zdt4"` | 10 | Multimodal landscape |
| `"zdt6"` | 10 | Non-uniform density |

**DTLZ suite (scalable objectives)**

| Name | Default objectives | Description |
|------|:-----------------:|-------------|
| `"dtlz1"` | 3 | Linear front, many local optima |
| `"dtlz2"` | 3 | Spherical front, standard test case |
| `"dtlz3"` | 3 | Spherical front, multimodal |
| `"dtlz4"` | 3 | Biased density |
| `"dtlz5"` | 3 | Degenerate front |
| `"dtlz6"` | 3 | Degenerate, multimodal |
| `"dtlz7"` | 3 | Disconnected front |

Override the number of objectives with `n_obj`:

```python
result = optimize("dtlz2", algorithm="nsgaiii", n_obj=5, max_evaluations=50000, seed=0)
```

---

## VAMOS Studio

Interactive Streamlit dashboard for visualization and exploration.

```bash
# Install
pip install "vamos-optimization[studio]"

# Launch
python -m vamos.studio
```

Features: problem builder, Pareto front plots, solution comparison, hypervolume tracking.

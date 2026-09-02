# Stable API overview

Import stable VAMOS 1.0.0 features from `vamos`, `vamos.api`,
`vamos.algorithms`, `vamos.problems`, `vamos.run_artifacts`, or
`vamos.study_artifacts`. Deep implementation modules are internal.

## `optimize()`

```text
result = optimize(
    problem,
    *,
    algorithm="auto",
    max_evaluations=None,
    termination=None,
    pop_size=None,
    engine=None,
    seed=42,
    verbose=False,
    n_var=None,
    n_obj=None,
    problem_kwargs=None,
    algorithm_config=None,
    eval_strategy=None,
    live_viz=None,
    checkpoint=None,
)
```

The common parameters are:

| Name | Meaning |
| --- | --- |
| `problem` | Registered problem name or a compatible problem instance. |
| `algorithm` | Built-in algorithm ID, plugin ID, or `"auto"`. |
| `max_evaluations` | Hard objective-evaluation budget. |
| `pop_size` | Convenience population-size override. |
| `engine` | Explicit computation backend, or `None` for the NumPy default. |
| `seed` | One integer/`None`, or a list/tuple for an in-memory multi-seed result. |
| `algorithm_config` | Public typed configuration for a fully specified algorithm. |
| `eval_strategy` | `"serial"`, `"multiprocessing"`, `"dask"`, or a backend object. |

One integer seed returns `OptimizationResult`; a list or tuple returns the
sequence-compatible `StudyResult`. A durable study uses `StudySpec` and
`create_study` instead.

## `make_problem()`

```text
problem = make_problem(
    fn,
    *,
    n_var,
    n_obj,
    encoding,
    bounds=None,
    xl=None,
    xu=None,
    vectorized=False,
    name=None,
    constraints=None,
    n_constraints=0,
)
```

`encoding` is required and is one of `"real"`, `"binary"`, `"integer"`,
`"permutation"`, or `"mixed"`. In scalar mode the callable receives one
solution. In vectorized mode it receives a batch. Constraint values use
`g(x) <= 0` for feasibility.

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=400, pop_size=40, seed=42)
```

## Results

`OptimizationResult.F` holds objective values and `OptimizationResult.X` holds
decision variables. `data` contains numerical run data and `meta` contains
metadata. Selection helpers include `front()`, `best()`, and `top_k()`.

`StudyResult` supports sequence access plus `metric_values()`, `mean()`,
`std()`, and `best_run()` for numeric run metrics.

## Canonical run lifecycle

```python
from vamos import load_result, load_run, reproduce, save_result, verify_run

stored = save_result(result, "runs/example")
run = load_run(stored.root)
loaded = load_result(stored.root)
verification = verify_run(stored.root, require_level="exact")
replay = reproduce(stored.root, output="runs/replays/example")
```

Loading and verification are inert. Replay is an explicit execution operation.

## Stability boundary

Studio, tuning, provider integrations, plugins, statistical analysis,
visualization, and non-stable CLI commands remain experimental in 1.0.0. See
the repository's canonical stability and known-limitations pages for the full
contract.

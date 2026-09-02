# Getting started

VAMOS 1.0.0 supports Python 3.10, 3.11, and 3.12 on Linux, and Python 3.12 on
Windows and macOS, as exercised by the release CI matrix.

## Install

```bash
python -m venv .venv
python -m pip install vamos-optimization
python -c "import vamos; print(vamos.__version__)"
vamos check
```

The version command must print `1.0.0`.

Optional extras are capability groups:

```bash
python -m pip install "vamos-optimization[compute]"  # Numba, MooCore, Dask
python -m pip install "vamos-optimization[analysis]" # plotting and notebooks
python -m pip install "vamos-optimization[tuning]"   # model-based tuning
python -m pip install "vamos-optimization[studio]"   # experimental Panel Studio
```

## First optimization

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)

print(result.F.shape)              # (n_solutions, 2)
print(result.X.shape)              # (n_solutions, 30)
print(result.data["evaluations"]) # 400
```

`F` and `X` are NumPy arrays. An explicit seed owns the stochastic path.
Same-environment determinism does not promise bitwise equality across backends
or platforms.

## Custom problem

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)

result = optimize(
    problem,
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    seed=42,
)
```

The default scalar adapter evaluates one solution at a time. Pass
`vectorized=True` to `make_problem` only when the callable accepts an
`(N, n_var)` batch and returns an `(N, n_obj)` array.

## Explicit configuration

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
config = NSGAIIConfig.default(pop_size=40, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=config,
    max_evaluations=400,
    seed=42,
)
```

Use top-level `pop_size` for ordinary runs and a public configuration object
when the algorithm configuration must be fully specified and preserved.

## Parallel evaluation

For an expensive custom objective, `eval_strategy="multiprocessing"` can
evaluate solutions through the optional compute stack. This is an evaluation
backend; it does not make durable study mutation multi-owner or distributed.

## Next steps

- [Quickstart tutorial](tutorials/quickstart.md)
- [Algorithms](algorithms/index.md)
- [API reference](api/index.md)

# Getting started

Install
-------

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install "vamos[compute,research,analysis,dev]"
```

Useful extras:

- `compute`: accelerated kernels + distributed eval (numba, moocore, dask)
- `research`: external baselines + benchmarks (pymoo, jmetalpy, pygmo)
- `analysis`: plotting + notebook deps (matplotlib/plotly/scikit-learn, ipywidgets, nbconvert)
- `tuning`: model-based hyperparameter tuning backends (Optuna, SMAC3, BOHB)
- `dev`: pytest, ruff, black, nbformat/nbconvert for notebook checks
- `examples`: minimal plotting + scikit-learn deps
- `studio`: streamlit + plotly dashboard
- `autodiff`: JAX for constraint autodiff helpers

Smoke tests
-----------

- Core check: `vamos check`
- Guided quickstart: `vamos quickstart` (use `--template list` to see domain templates)
- Quick NSGA-II run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
- Full test suite (core): `pytest`
- With extras installed: `pytest -m "not slow"`
- List all subcommands: `vamos help`
- If you hit missing-dependency or unknown-key errors, see `docs/guide/troubleshooting.md`.

Interactive tutorial
--------------------

For a hands-on walkthrough with runnable code, open the interactive tutorial notebook:

```bash
jupyter notebook notebooks/0_basic/05_interactive_tutorial.ipynb
```

It covers installation verification, first optimization, custom problems, algorithm
comparison, constraints, parameter tuning, and exporting results for papers.

Python API
----------

Preferred path: start with `optimize(...)`. Reach for config objects only when you need fully specified, reproducible runs or plugin algorithms.
If you are new to Python, start with `docs/guide/minimal-python.md`.
For a quick comparison, see `notebooks/0_basic/00_api_comparison.ipynb`.

**1. One-liner (Unified API):**

```python
from vamos import optimize
from vamos.ux.api import result_summary_text

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10_000, pop_size=100, seed=42, verbose=True)
print(result_summary_text(result))
```

**2. Your own problem (no class needed):**

```python
from vamos import make_problem, optimize

# Write a simple function -- VAMOS handles vectorization and the protocol
problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2,
    bounds=[(0, 1), (0, 1)],
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
```

Or scaffold a file interactively: `vamos create-problem`.

**3. Advanced control (explicit args + config objects):**

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    termination=("max_evaluations", 10_000),
    seed=42,
    engine="numpy",
)
```

Prefer the unified `optimize(...)` API; use algorithm config objects for reproducible, fully specified runs. For plugin algorithms, wrap free-form mappings in `GenericAlgorithmConfig`.

API decision guide
------------------

Use the lightest interface that still makes the run reproducible.

| Goal | Use | Example |
| --- | --- | --- |
| Quick scripts, notebooks | Unified `optimize(...)` | `optimize("zdt1", algorithm="nsgaii", max_evaluations=5000)` |
| Your own problem | `make_problem(fn, ...)` | `make_problem(my_fn, n_var=2, n_obj=2, bounds=[(0,1),(0,1)])` |
| Scaffold a problem file | CLI wizard | `vamos create-problem` |
| Reproducible configs | `algorithm_config` (via `.default()` or `.builder()`) + explicit termination | `optimize(problem, algorithm="nsgaii", algorithm_config=cfg, termination=("max_evaluations", 5000))` |
| Plugin algorithms | `GenericAlgorithmConfig` | `optimize(problem, algorithm="my_algo", algorithm_config=GenericAlgorithmConfig({...}))` |
| Small study in one call | `seed=[...]` | `optimize("zdt1", seed=[0, 1, 2])` |

Benchmarks and studies
----------------------

- Compare backends: `python -m vamos.experiment.cli.main --experiment backends --problem zdt1`
- Run a predefined suite: `vamos bench --suite ZDT_small --algorithms nsgaii moead --output report/`
- Batch problem x algorithm sweeps: `vamos --problem-set families --algorithm both`

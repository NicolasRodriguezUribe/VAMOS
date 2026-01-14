# Getting started

Install
-------

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[compute,research,dev]"
```

Useful extras:

- `compute`: accelerated kernels + distributed eval (numba, moocore, dask)
- `research`: external baselines + benchmarks (pymoo, jmetalpy, pygmo)
- `analysis`: plotting + notebook deps (matplotlib/plotly/scikit-learn, ipywidgets, nbconvert)
- `dev`: pytest, ruff, black, nbformat/nbconvert for notebook checks
- `examples`: minimal plotting + scikit-learn deps
- `studio`: streamlit + plotly dashboard
- `autodiff`: JAX for constraint autodiff helpers

Smoke tests
-----------

- Core check: `python -m vamos.experiment.diagnostics.self_check` or `vamos-self-check`
- Quick NSGA-II run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
- Full test suite (core): `pytest`
- With extras installed: `pytest -m "not slow"`
- If you hit missing-dependency or unknown-key errors, see `docs/guide/troubleshooting.md`.

Python API
----------

**1. One-liner (Unified API):**

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", budget=10_000, pop_size=100, seed=42, verbose=True)
print(result.summary_text())
print(result.explain_defaults())
```

**2. Full control (explicit args + config objects):**

```python
from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

problem = make_problem_selection("zdt1", n_var=30).instantiate()
algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    termination=("n_eval", 10_000),
    seed=42,
    engine="numpy",
)
```

Prefer the unified `optimize(...)` API; use algorithm config objects for reproducible, fully-specified runs. For plugin algorithms, wrap free-form mappings in `GenericAlgorithmConfig`.

Benchmarks and studies
----------------------

- Compare backends: `python -m vamos.experiment.cli.main --experiment backends --problem zdt1`
- Run a predefined suite: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/`
- Batch problem x algorithm sweeps: `vamos --problem-set families --algorithm both`


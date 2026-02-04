# Getting started

Install
-------

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install "vamos[compute,research,dev]"
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
- Guided quickstart: `vamos quickstart` (use `--template list` to see domain templates)
- Quick NSGA-II run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
- Full test suite (core): `pytest`
- With extras installed: `pytest -m "not slow"`
- If you hit missing-dependency or unknown-key errors, see `docs/guide/troubleshooting.md`.

Python API
----------

Preferred path: start with `optimize(...)`. Reach for config objects only when you need fully specified, reproducible runs or plugin algorithms.
If you are new to Python, start with `docs/guide/minimal-python.md`.
For a quick comparison, see `notebooks/0_basic/00_api_comparison.ipynb`.

**1. One-liner (Unified API):**

```python
from vamos import optimize
from vamos.ux.api import result_summary_text

result = optimize("zdt1", algorithm="nsgaii", budget=10_000, pop_size=100, seed=42, verbose=True)
print(result_summary_text(result))
```

**2. Advanced control (explicit args + config objects):**

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
    termination=("n_eval", 10_000),
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
| Quick scripts, notebooks | Unified `optimize(...)` | `optimize("zdt1", algorithm="nsgaii", budget=5000)` |
| Reproducible configs | `algorithm_config` (via `.default()` or `.builder()`) + explicit termination | `optimize(problem, algorithm="nsgaii", algorithm_config=cfg, termination=("n_eval", 5000))` |
| Plugin algorithms | `GenericAlgorithmConfig` | `optimize(problem, algorithm="my_algo", algorithm_config=GenericAlgorithmConfig({...}))` |
| Small study in one call | `seed=[...]` | `optimize("zdt1", seed=[0, 1, 2])` |

Benchmarks and studies
----------------------

- Compare backends: `python -m vamos.experiment.cli.main --experiment backends --problem zdt1`
- Run a predefined suite: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/`
- Batch problem x algorithm sweeps: `vamos --problem-set families --algorithm both`


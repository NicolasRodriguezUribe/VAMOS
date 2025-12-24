# Getting started

Install
-------

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[backends,benchmarks,dev]"
```

Useful extras:

- `backends`: numba and moocore kernels
- `benchmarks`: pymoo, jmetalpy, pygmo baselines
- `dev`: pytest, ruff, black, nbformat/nbconvert for notebook checks
- `notebooks` / `examples`: plotting and scikit-learn deps (interactive explorer also needs plotly + ipywidgets, included in `notebooks`)
- `studio`: streamlit + plotly dashboard
- `autodiff`: JAX for constraint autodiff helpers

Smoke tests
-----------

- Core check: `python -m vamos.experiment.diagnostics.self_check` or `vamos-self-check`
- Quick NSGA-II run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
- Full test suite (core): `pytest`
- With extras installed: `pytest -m "not slow"`

Benchmarks and studies
----------------------

- Compare backends: `python -m vamos.experiment.cli.main --experiment backends --problem zdt1`
- Run a predefined suite: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/`
- Batch problem x algorithm x seed sweeps: `python -m vamos.experiment.study.runner --suite families`

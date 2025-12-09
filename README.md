# VAMOS - Vectorized Architecture for Multiobjective Optimization Studies

![VAMOS](VAMOS.jpeg)

Minimal steps to run and explore:

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install everything you typically need (core + optional backends + dev/test)
pip install -e ".[backends,benchmarks,dev]"

# 3) Run a quick NSGA-II smoke on ZDT1 (NumPy backend)
python -m vamos.cli.main --problem zdt1 --max-evaluations 2000

# 4) Try the MooCore backend
python -m vamos.cli.main --engine moocore --problem zdt1 --max-evaluations 2000

# 5) Run tests
pytest
```

## What's inside

- Algorithms: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO with vectorized kernels (NumPy, Numba, MooCore).
- Encodings: continuous, permutation, binary, integer, mixed.
- Config + tuning: hierarchical `AlgorithmConfigSpace`, meta-level NSGA-II tuner, templates, multi-algorithm search. CLI/runner can read YAML/JSON specs (see below).
- Analysis: constraint handling toolbox, objective reduction, MCDM helpers, visualization utilities, stats (Friedman/Wilcoxon/CD plots).
- Problems: ZDT, DTLZ, WFG, LZ09 (pymoo-backed), CEC2009 UF/CF (pymoo-backed), TSP/TSPLIB, binary (feature selection/knapsack/QUBO), integer (allocation/job assignment), mixed design, welded beam and real-data examples (SVM tuning, feature selection).
- CLI/runner: experiment orchestration, metrics, result CSVs/metadata under `results/`.
- StudyRunner: batch problem x algorithm x seed sweeps.
- Notebooks: exploratory examples under `notebooks/` (optional).

## Common commands

- Run with different algorithms/problems:
  - `python -m vamos.cli.main --algorithm moead --problem dtlz2 --n-obj 3`
  - `python -m vamos.cli.main --algorithm spea2 --problem zdt1 --max-evaluations 1000`
- `python -m vamos.cli.main --problem-set families`
- `python -m vamos.cli.main --problem tsp6`
- Use hypervolume early-stop on NSGA-II:
  - `python -m vamos.cli.main --hv-threshold 0.5 --hv-reference-front data/reference_fronts/ZDT1.csv`
- Run tuning (outer NSGA-II over hyperparameters) via code (see `vamos.tuning`).
- Visualize or post-hoc analyze:
  - `from vamos.visualization import plot_pareto_front_2d`
  - `from vamos.analysis.mcdm import weighted_sum_scores`
  - `from vamos.analysis.stats import friedman_test, plot_critical_distance`
- Quick sanity check of your install:
  - `python -m vamos.diagnostics.self_check`
  - or `vamos-self-check`

## Migration note

- Deprecated tuning shim modules (e.g., `vamos.tuning.param_space`, `vamos.tuning.parameter_space`, `vamos.tuning.random_search_tuner`, `vamos.tuning.meta`, etc.) have been removed. Import directly from the canonical subpackages: `vamos.tuning.core.*`, `vamos.tuning.evolver.*`, and `vamos.tuning.racing.*`.

## Documentation

- Browse the docs under `docs/` (MkDocs). Key pages cover CLI/config, algorithms/backends, problems, constraint DSL/autodiff, and extension guides.
- Build locally: `mkdocs serve` (or `py -m mkdocs serve`) after installing the `docs` extra with `pip install -e ".[docs]"`.

### Examples

- End-to-end scripts in `examples/`: `hyperparam_tuning.py`, `feature_selection.py`, `engineering_design.py`.
- Enable optional deps with `pip install -e ".[examples]"` to run scikit-learn based examples and plots.

## Contributing

- See `CONTRIBUTING.md` for how to add problems, algorithms, or backends.
- Prefer adding type hints/docstrings on public APIs; keep performance-critical kernels unchanged unless benchmarked.

## Config files (YAML/JSON)

You can provide experiment defaults via `--config path/to/spec.yaml`. Example:

```yaml
defaults:
  algorithm: moead
  engine: numpy
  population_size: 120
  max_evaluations: 20000
  hv_threshold: 0.8
  moead:
    crossover: {method: sbx, prob: 0.9, eta: 20}
    mutation: {method: pm, prob: "1/n", eta: 20}
problems:
  bin_knapsack:
    algorithm: nsgaii
    n_var: 30
    population_size: 150
  nsgaii:
    crossover: {method: uniform}
    mutation: {method: bitflip, prob: "1/n"}
```

Quick presets for the new algorithms:

```yaml
spea2:
  pop_size: 80
  archive_size: 80
  crossover: {method: sbx, prob: 0.9, eta: 20}
  mutation: {method: pm, prob: "1/n", eta: 20}
ibea:
  indicator: eps  # or hypervolume
  kappa: 0.05
  crossover: {method: sbx, prob: 0.9, eta: 20}
  mutation: {method: pm, prob: "1/n", eta: 20}
smpso:
  pop_size: 80
  archive_size: 80
  inertia: 0.5
  c1: 1.5
  c2: 1.5
  vmax_fraction: 0.5
  mutation: {method: pm, prob: "1/n", eta: 20}
```

CLI flags override config values. Per-problem sections override defaults.

## Performance toggle

Enable optional Numba-accelerated variation for permutation/binary/integer encodings:

```bash
export VAMOS_USE_NUMBA_VARIATION=1  # set before running
```

## Package layout (src/vamos)

- `algorithm/` - evolutionary cores and config builders.
- `kernel/` - NumPy/Numba/MooCore kernels.
- `problem/` - benchmark problems and registry.
- `tuning/` - meta-optimizer (outer NSGA-II), config spaces, pipelines.
- `constraints.py` - feasibility/penalty strategies.
- `objective_reduction.py` - correlation/angle/hybrid reducers.
- `mcdm.py`, `visualization.py`, `stats.py` - decision-making, plotting, post-hoc stats.
- `runner.py`, `cli.py`, `study/` - CLI and study orchestration.

## Dependencies

### Core dependencies (always installed)
- `numpy>=1.23`
- `scipy>=1.10`

### Optional extras

Install with `pip install -e ".[extra1,extra2]"`:

| Extra        | Packages                                      | Purpose                          |
|--------------|-----------------------------------------------|----------------------------------|
| `backends`   | `numba>=0.57`, `moocore>=0.4`                 | Accelerated backends             |
| `benchmarks` | `pymoo>=0.6`, `jmetalpy>=1.5`, `pygmo>=2.19`  | Benchmark comparisons            |
| `dev`        | `pytest>=7.0`, `black>=23.0`, `ruff>=0.1.5`   | Development & testing            |
| `notebooks`  | `pandas>=1.5`, `matplotlib>=3.7`, `seaborn>=0.12`, `ipython>=8.10`, `scikit-learn>=1.3` | Jupyter notebook support     |
| `examples`   | `pandas>=1.5`, `matplotlib>=3.7`, `seaborn>=0.12`, `scikit-learn>=1.3` | Run the real-data examples   |

### Quick install examples

```powershell
# Full install (all extras)
pip install -e ".[backends,benchmarks,dev,notebooks]"

# Minimal install for notebooks
pip install -e ".[backends,notebooks]"

# Just core + moocore backend
pip install -e ".[backends]"
```

## Additional benchmarks & examples

- New benchmark families: LZ09 (F1-F9, built-in) and CEC2009 UF1-UF3 + CF1 (fallback implementations shipped; will use pymoo if installed).
- Real-world problems: mixed hyperparameter tuning (`ml_tuning`), welded beam design (`welded_beam`), and binary feature selection (`fs_real`) are registered and usable via CLI or `optimize()`.
- Example scripts under `examples/` (run with `pip install -e ".[examples]"`):
  - `python examples/hyperparam_tuning_pipeline.py`
  - `python examples/engineering_design_pipeline.py`
  - `python examples/feature_selection_qubo.py`
- Paper-ready benchmarking: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/` runs predefined suites, writes raw runs + summary CSVs + LaTeX-ready tables and plots under `report/`.
- Interactive decision-making: install `pip install -e ".[studio]"` and run `vamos-studio --study-dir results` to explore fronts, rank with preferences, inspect solutions, export, and trigger focused follow-up runs.
- Adaptive hyper-heuristics: NSGA-II can enable online operator portfolios (bandit-based epsilon-greedy/UCB) via `adaptive_operators` in the config; portfolio utilities live under `vamos.hyperheuristics`.
- Notebooks & examples: install `pip install -e ".[notebooks]"` and open the notebooks folder for runnable quickstarts (`00_quickstart_vamos.ipynb`, `01_benchmarks_and_metrics.ipynb`, `02_tuning_and_metaoptimization.ipynb`).

## Testing & QA

- Core tests: `pytest`
- Full optional stack (after `pip install -e ".[dev,backends,benchmarks,studio,analytics,autodiff,notebooks]"`): `pytest -m "not slow"`
- Examples/notebooks: set `VAMOS_RUN_NOTEBOOK_TESTS=1` then `pytest -m "examples"`

## Notes

- Dependencies are declared in `pyproject.toml`; extras `[backends]`, `[benchmarks]`, `[dev]`, `[notebooks]` cover typical setups.
- Results live under `results/` by default (`FUN.csv`, `metadata.json`, etc.).
- Tests are under `tests/` and follow AAA style. Run `pytest` after installing with `[dev]`.

# VAMOS – Vectorized Architecture for Multiobjective Optimization Studies

![VAMOS](VAMOS.jpeg)

Minimal steps to run and explore:

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install everything you typically need (core + optional backends + dev/test)
pip install -e ".[backends,benchmarks,dev]"

# 3) Run a quick NSGA-II smoke on ZDT1 (NumPy backend)
python -m vamos.main --problem zdt1 --max-evaluations 2000

# 4) Try the MooCore backend
python -m vamos.main --engine moocore --problem zdt1 --max-evaluations 2000

# 5) Run tests
pytest
```

## What’s inside

- Algorithms: NSGA-II/III, MOEA/D, SMS-EMOA with vectorized kernels (NumPy, Numba, MooCore).
- Encodings: continuous, permutation, binary, integer, mixed.
- Config + tuning: hierarchical `AlgorithmConfigSpace`, meta-level NSGA-II tuner, templates, multi-algorithm search. CLI/runner can read YAML/JSON specs (see below).
- Analysis: constraint handling toolbox, objective reduction, MCDM helpers, visualization utilities, stats (Friedman/Wilcoxon/CD plots).
- Problems: ZDT, DTLZ, WFG, TSP/TSPLIB, binary (feature selection/knapsack/QUBO), integer (allocation/job assignment), mixed design.
- CLI/runner: experiment orchestration, metrics, result CSVs/metadata under `results/`.
- StudyRunner: batch problem × algorithm × seed sweeps.
- Notebooks: exploratory examples under `notebooks/` (optional).

## Common commands

- Run with different algorithms/problems:
  - `python -m vamos.main --algorithm moead --problem dtlz2 --n-obj 3`
  - `python -m vamos.main --problem-set families`
  - `python -m vamos.main --problem tsp6`
- Use hypervolume early-stop on NSGA-II:
  - `python -m vamos.main --hv-threshold 0.5 --hv-reference-front data/reference_fronts/ZDT1.csv`
- Run tuning (outer NSGA-II over hyperparameters) via code (see `vamos.tuning`).
- Visualize or post-hoc analyze:
  - `from vamos.visualization import plot_pareto_front_2d`
  - `from vamos.mcdm import weighted_sum_scores`
  - `from vamos.stats import friedman_test, plot_critical_distance`
- Quick sanity check of your install:
  - `python -m vamos.self_check`
  - or `vamos-self-check`

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

CLI flags override config values. Per-problem sections override defaults.

## Performance toggle

Enable optional Numba-accelerated variation for permutation/binary/integer encodings:

```bash
export VAMOS_USE_NUMBA_VARIATION=1  # set before running
```

## Package layout (src/vamos)

- `algorithm/` – evolutionary cores and config builders.
- `kernel/` – NumPy/Numba/MooCore kernels.
- `problem/` – benchmark problems and registry.
- `tuning/` – meta-optimizer (outer NSGA-II), config spaces, pipelines.
- `constraints.py` – feasibility/penalty strategies.
- `objective_reduction.py` – correlation/angle/hybrid reducers.
- `mcdm.py`, `visualization.py`, `stats.py` – decision-making, plotting, post-hoc stats.
- `runner.py`, `cli.py`, `study/` – CLI and study orchestration.

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
| `notebooks`  | `pandas>=1.5`, `matplotlib>=3.7`, `ipython>=8.10` | Jupyter notebook support     |

### Quick install examples

```powershell
# Full install (all extras)
pip install -e ".[backends,benchmarks,dev,notebooks]"

# Minimal install for notebooks
pip install -e ".[backends,notebooks]"

# Just core + moocore backend
pip install -e ".[backends]"
```

## Notes

- Dependencies are declared in `pyproject.toml`; extras `[backends]`, `[benchmarks]`, `[dev]`, `[notebooks]` cover typical setups.
- Results live under `results/` by default (`FUN.csv`, `metadata.json`, etc.).
- Tests are under `tests/` and follow AAA style. Run `pytest` after installing with `[dev]`.

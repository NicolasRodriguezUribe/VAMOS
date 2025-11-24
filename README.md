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
- Config + tuning: hierarchical `AlgorithmConfigSpace`, meta-level NSGA-II tuner, templates, multi-algorithm search.
- Analysis: constraint handling toolbox, objective reduction, MCDM helpers, visualization utilities, stats (Friedman/Wilcoxon/CD plots).
- Problems: ZDT, DTLZ, WFG, TSP/TSPLIB.
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

## Package layout (src/vamos)

- `algorithm/` – evolutionary cores and config builders.
- `kernel/` – NumPy/Numba/MooCore kernels.
- `problem/` – benchmark problems and registry.
- `tuning/` – meta-optimizer (outer NSGA-II), config spaces, pipelines.
- `constraints.py` – feasibility/penalty strategies.
- `objective_reduction.py` – correlation/angle/hybrid reducers.
- `mcdm.py`, `visualization.py`, `stats.py` – decision-making, plotting, post-hoc stats.
- `runner.py`, `cli.py`, `study/` – CLI and study orchestration.

## Notes

- Dependencies are declared in `pyproject.toml`; extras `[backends]`, `[benchmarks]`, `[dev]` cover typical setups.
- Results live under `results/` by default (`FUN.csv`, `metadata.json`, etc.).
- Tests are under `tests/` and follow AAA style. Run `pytest` after installing with `[dev]`.

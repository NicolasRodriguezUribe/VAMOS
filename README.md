# VAMOS - Vectorized Architecture for Multiobjective Optimization Studies

VAMOS is a compact research playground for NSGA-II/III, MOEA/D, and SMS-EMOA with multiple vectorized backends (NumPy, Numba, and MooCore). The codebase is laid out to stay easy to install, easy to navigate, and friendly for experimentation.

```
.
|-- build/          # generated weight files or misc artifacts
|-- docs/           # architecture notes or API docs
|-- notebooks/      # exploratory notebooks (not part of the package)
|-- results/        # FUN.csv and metadata from CLI runs
|-- src/
|   `-- vamos/      # installable Python package (algorithms, kernels, CLI)
|-- tests/          # pytest-based smoke/benchmark suites
|-- pyproject.toml  # package + dependency metadata
`-- README.md
```

## Quick start (PowerShell)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package in editable mode (core dependencies only)
pip install -e .

# 3. Optional extras:
#    - backends: numba + moocore
#    - benchmarks: pymoo + jmetalpy + pygmo
pip install -e .[backends]
pip install -e .[benchmarks]

# 4. Run the CLI (default NSGA-II / NumPy backend / ZDT1)
python -m vamos.main

# 5. Try other combinations
python -m vamos.main --engine moocore_v2
python -m vamos.main --algorithm moead --problem dtlz2 --n-obj 3
python -m vamos.main --problem-set families  # ZDT1 + DTLZ2 + WFG4 (WFG requires pymoo)
python -m vamos.main --include-external --external-problem-source vamos
python -m vamos.main --problem tsp6  # Toy permutation-based TSP instance
python -m vamos.main --problem kroa100  # TSPLIB permutation benchmark (100 cities)
python -m vamos.main --experiment backends --include-external
python -m vamos.main --problem-set tsplib_kro100  # KroA-E100 sweep
```

### NSGA-II variation flags

Continuous NSGA-II runs expose the crossover, mutation, and repair operators via CLI flags:

- `--nsgaii-crossover {sbx,blx_alpha}` with optional `--nsgaii-crossover-prob`, `--nsgaii-crossover-eta`, or `--nsgaii-crossover-alpha`.
- `--nsgaii-mutation {pm,non_uniform}` with optional `--nsgaii-mutation-prob`, `--nsgaii-mutation-eta`, and `--nsgaii-mutation-perturbation`.
- `--nsgaii-repair {clip,reflect,resample,random,round,none}` to control how offspring are projected back into bounds (use `round` to snap to the nearest integer before clamping).
- `--population-size` sets the working population for every internal algorithm; `--offspring-population-size` (even integer) controls how many solutions NSGA-II generates per generation so you can mimic settings like lambda=56 / mu=14.
- `--hv-threshold` (with an optional `--hv-reference-front`) lets NSGA-II stop automatically once the hypervolume reaches a given fraction of a reference Pareto front. Built-in fronts for ZDT1/2/3/4/6 live under `data/reference_fronts/`. (Hypervolume-based early stop is currently available for NSGA-II only.)

A quick smoke test for the BLX + non-uniform combination:

```powershell
python -m vamos.main --problem zdt1 --max-evaluations 2000 `
  --nsgaii-crossover blx_alpha --nsgaii-crossover-alpha 0.35 --nsgaii-crossover-prob 0.85 `
  --nsgaii-mutation non_uniform --nsgaii-mutation-perturbation 0.4 --nsgaii-mutation-prob 1/n `
  --nsgaii-repair reflect
```

When installing in editable mode you also get the `vamos` console script, so after step 3 you can run `vamos --experiment backends`.

## Working with the code

### Package layout (`src/vamos`)

- `main.py`: CLI entry point that wires parsing + runner.
- `cli.py`: argument parsing helpers.
- `runner.py`: orchestration of experiments and metric computation.
- `external.py`: third-party baselines (PyMOO, jMetalPy, PyGMO).
- `plotting.py`: Pareto front plotting helpers.
- `algorithm/`: evolutionary cores and configuration builders.
- `kernel/`: vectorized operator implementations (NumPy, Numba, MooCore).
- `problem/`: benchmark definitions (ZDT, DTLZ, WFG, TSP/TSPLIB).
- `study/`: helper utilities for scripted experiments/diagnostics.

Because everything lives under `src/vamos`, imports stay clean (`from vamos.algorithm import ...`) and tools like pytest/ruff work without additional `PYTHONPATH` tweaks.

### Results

Every CLI run creates `results/<PROBLEM>/<ALGO>/<ENGINE>/seed_<N>/` containing:

- `FUN.csv` - final objective vectors
- `time.txt` - elapsed time in ms
- `metadata.json` - problem/back-end configuration

Summary tables show hypervolume values computed with a shared reference point.

### Tests

Add pytest modules under `tests/`. Pyproject already hooks setuptools to look at `src/`, so `pytest` automatically discovers the package.

```powershell
pip install -e .[dev]
pytest
```

## Optional backends and baselines

| Feature            | Extra                              |
|--------------------|------------------------------------|
| `--engine numba`   | `pip install -e .[backends]`       |
| `--engine moocore` | same as above (requires the wheel) |
| PyMOO baseline     | `pip install -e .[benchmarks]`     |
| jMetalPy baseline  | same as above                      |
| PyGMO baseline     | install PyGMO separately (conda)   |

External baselines use each library's native benchmark definitions by default. Pass `--external-problem-source vamos` to wrap the VAMOS problem implementations instead (currently available for ZDT1).

Only the dependencies you install are loaded; missing libraries are skipped gracefully with a console warning.

### Permutation / discrete problems

The framework understands permutation encodings for NSGA-II. Besides the toy `tsp6` benchmark we vendor TSPLIB's KroA/B/C/D/E100 instances (`--problem kroa100`, etc.), each exposing the same bi-objective TSP (minimize tour length and maximum edge). When selecting permutation problems the CLI automatically switches NSGA-II to order crossover + swap mutation while keeping the rest of the workflow (kernels, diagnostics, plotting) unchanged.

## Contributing / Extending

- Create new problems under `src/vamos/problem/` and register them in `problem/registry.py`.
- Write new kernels by subclassing `KernelBackend` in `src/vamos/kernel/backend.py` and add a branch in `_resolve_kernel`.
- Use `docs/` for design notes and `notebooks/` for exploratory work—the `src/` package stays focused on installable code.

Happy optimizing!

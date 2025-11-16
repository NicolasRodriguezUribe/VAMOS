# VAMOS – Vectorized Architecture for Multiobjective Optimization Studies

VAMOS is a lean research-oriented implementation of NSGA-II, MOEA/D, and SMS-EMOA tailored for experimentation with vectorized evolutionary kernels. The project focuses on keeping individuals in contiguous NumPy arrays (Structure-of-Arrays style) so that core operators can be swapped easily between backends such as pure NumPy, Numba-accelerated kernels, or the C-based [moocore](https://pypi.org/project/moocore/).

## Goals

- Provide fully vectorized NSGA-II, MOEA/D, and SMS-EMOA kernels for benchmarking and teaching.
- Offer multiple backends so you can compare CPU-oriented implementations:
  - `numpy`: pure NumPy operators, easy to understand and extend.
  - `numba`: ranking/survival routines compiled with Numba for better CPU throughput.
  - `moocore`: hybrid backend that delegates non-dominated sorting to moocore’s C routines.
  - `moocore_v2`: moves additional survival logic (rank filtering + hypervolume tie-breaks) into moocore for maximum CPU efficiency.
- Keep the code small and self-contained for experimentation with new mutation/crossover operators or problem definitions.

## Project Layout

```
.
├── algorithm/
│   ├── config.py           # Declarative configuration builders
│   ├── nsgaii.py           # Vectorized NSGA-II evolutionary loop
│   ├── moead.py            # Vectorized MOEA/D evolutionary loop
│   ├── smsemoa.py          # SMS-EMOA implementation
│   ├── weight_vectors.py   # Simplex-lattice weight utilities
│   └── hypervolume.py      # Low-dimensional hypervolume helpers
├── kernel/
│   ├── numpy_backend.py    # Reference pure NumPy kernel implementations
│   ├── numba_backend.py    # Numba-accelerated ranking/survival
│   └── moocore_backend.py  # Backends that leverage moocore for ranking/survival
├── problem/
│   └── zdt1.py             # ZDT1 test problem
├── results/                # Output directory (created automatically)
└── main.py                 # CLI entry point
`


## Requirements

- Python 3.12 (project uses a virtual environment under `.venv/`)
- NumPy (installed in the virtual environment)
- Optional dependencies:
  - `numba` (for the Numba backend)
  - `moocore` (for moocore-based backends)

Install dependencies inside the provided venv, e.g.:

```powershell
.\.venv\Scripts\pip install numba moocore
```

## Running VAMOS

Use the CLI in `main.py`. The most common options:

```powershell
# Run once with the default backend (numpy)
.\.venv\Scripts\python.exe main.py

# Choose a backend explicitly
.\.venv\Scripts\python.exe main.py --engine moocore_v2

# Switch algorithms (NSGA-II, MOEA/D, or SMS-EMOA)
.\.venv\Scripts\python.exe main.py --algorithm moead --engine numpy

# Run SMS-EMOA with the NumPy backend
.\.venv\Scripts\python.exe main.py --algorithm smsemoa --engine numpy

# Solve with every algorithm sequentially (per chosen backend/experiment)
.\.venv\Scripts\python.exe main.py --algorithm both

# Benchmark every backend sequentially
.\.venv\Scripts\python.exe main.py --experiment backends
```

Configuration parameters (population size, max evaluations, number of variables, etc.) are defined at the top of `main.py`. The configuration builders in `algorithm/config.py` (`NSGAIIConfig`, `MOEADConfig`, and `SMSEMOAConfig`) make it easy to experiment with different crossover/mutation settings, decomposition neighborhoods, or SMS-EMOA reference-point setups.

When running MOEA/D, decomposition weight vectors are loaded from `build/weights/*.csv`. If the requested file is missing, the CLI automatically generates a simplex-lattice design with enough vectors for the configured population size and writes it to disk for future runs.

SMS-EMOA relies on low-dimensional hypervolume contributions (up to three objectives) together with an adaptive reference point. Override the `.reference_point()` builder options if your objective scales require a custom reference vector.

## Output

Each run prints a detailed log:

- Problem metadata (decision variables, objectives, backend name)
- Performance metrics (total time, evaluations per second, final population size)
- Objective ranges and, for bi-objective problems, approximate front spread

Results are saved to `results/VAMOS_ZDT1/`:

- `FUN.csv`: final objective values for each solution
- `time.txt`: elapsed time in milliseconds

When running `--experiment backends` an additional table summarises time, evaluations/sec, and spread for each backend.

## Extending the Project

- **New problem definitions**: add a module under `problem/`, implement `evaluate`, and plug it into `main.py`.
- **Custom backends**: create a class with the same interface as `NumPyKernel` (methods for `nsga2_ranking`, `tournament_selection`, `sbx_crossover`, `polynomial_mutation`, `nsga2_survival`) and add a resolver branch in `_resolve_kernel`.
- **Alternative operators**: adjust `kernel/numpy_backend.py` or implement them in a dedicated backend – the whole algorithm keeps individuals as NumPy arrays, so vectorized operations are straightforward.

## License

This project is distributed without an explicit license in the repository. Adapt or extend it according to your internal guidelines.

Happy optimizing with VAMOS!

# VAMOS overview

VAMOS (Vectorized Architecture for Multiobjective Optimization Studies) ships vectorized evolutionary algorithms, multiple kernels, benchmark suites, and orchestration tools for experiments and analysis.

- Algorithms: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO with continuous, permutation, binary, integer, and mixed encodings.
- Backends: NumPy (default), Numba, and MooCore kernels.
- Problems: ZDT, DTLZ, WFG, LZ09, CEC2009 UF/CF, TSP/TSPLIB, binary, integer, mixed, and real-data examples.
- Tooling: CLI runner, study runner, tuning/meta-optimization, benchmarking CLI, self-check, live visualization, and Studio (optional).

Quick start:

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e ".[backends,benchmarks,dev]"
python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000
```

Docs roadmap:
- `Getting Started` for install and smoke tests
- `CLI & Config` for flags and YAML/JSON specs
- `Algorithms & Backends` for capabilities and extras
- `Problems` for the registry and encodings
- `Constraints & Autodiff` for the DSL and JAX helpers
- `Extending VAMOS` for adding algorithms, operators, problems, or kernels

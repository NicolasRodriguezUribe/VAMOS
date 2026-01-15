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
pip install -e ".[compute,research,dev]"
python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000
```

New to Python? Start here:
- Minimal Python Track: `docs/guide/minimal-python.md`
- Guided wizard: `vamos quickstart`

Docs roadmap:
- `Guide`: Getting Started, CLI, Studio, Cookbook, Troubleshooting
- `Reference`: API docs, algorithms, problems, and constraints
- `Topics`: Hyperparameter tuning, analysis, extending VAMOS, and engineering details
- `Examples`: Comprehensive notebook suite (Basic, Intermediate, Advanced)

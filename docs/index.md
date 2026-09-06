# VAMOS overview

VAMOS (Vectorized Architecture for Multiobjective Optimization Studies) ships vectorized evolutionary algorithms, multiple kernels, benchmark suites, and orchestration tools for experiments and analysis.

VAMOS 1.0 distinguishes a stable optimization, run-artifact, and single-owner
study surface from experimental features such as Studio, provider integrations,
and tuning. See [Stability and versioning](project/stability-and-versioning.md)
before depending on an API as a 1.x compatibility commitment.

For citation metadata, see [CITATION.cff](https://github.com/vamos-optimization/VAMOS/blob/main/CITATION.cff).
Read the [security policy](https://github.com/vamos-optimization/VAMOS/blob/main/SECURITY.md)
and use [private vulnerability reporting](https://github.com/vamos-optimization/VAMOS/security/advisories/new)
to report a security issue.

- Algorithms: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA with continuous, permutation, binary, integer, and mixed encodings.
- Backends: NumPy (default exact reference), Numba for accelerated core kernels, and MooCore for indicator acceleration.
- Problems: ZDT, DTLZ, WFG, LZ09, CEC2009 UF/CF, TSP/TSPLIB, binary, integer, mixed, and real-data examples.
- Tooling: CLI runner, study runner, tuning/meta-optimization, benchmarking CLI, self-check, live visualization, and Studio (optional).

Quick start:

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install vamos-optimization
python -c "from vamos import optimize; result = optimize('zdt1', algorithm='nsgaii', max_evaluations=200, pop_size=40, seed=42); print(result.F.shape)"
```

Prefer this Python API path for the quickest first script. The stable
run-oriented and study CLI surfaces are listed in the
[stability policy](project/stability-and-versioning.md).

Optional model-based tuning backends (`optuna`, `bohb_optuna`, `smac3`, `bohb`):

```bash
pip install -e ".[tuning]"
```

New to Python? Start here:
- Minimal Python Track: `docs/guide/minimal-python.md`
- Installation: `docs/guide/installation.md`
- Durable studies: `docs/guide/studies.md`
- Guided wizard (experimental): `vamos quickstart`
- Customization and plugins: `docs/topics/extending.md`

Docs roadmap:
- `Guide`: Getting Started, CLI, Studio, Cookbook, Troubleshooting
- `Reference`: API docs, algorithms, problems, and constraints
- `Topics`: Hyperparameter tuning, analysis, extending VAMOS, and engineering details
- `Examples`: Comprehensive notebook suite (Basic, Intermediate, Advanced)

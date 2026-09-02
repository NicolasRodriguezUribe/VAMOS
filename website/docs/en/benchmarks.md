# Benchmark methodology

Runtime and solution-quality results are meaningful only when the compared
implementations have aligned semantics and the environment is recorded. VAMOS
does not claim a universal speedup over another framework.

## Record before comparing

- VAMOS and comparator versions, Python, NumPy, operating system,
  architecture, CPU, BLAS, and thread controls;
- problem dimensions, encoding, constraints, and objective implementation;
- population/offspring sizes, operators, stopping rule, and exact evaluation
  accounting;
- seed ownership and independent seed list;
- warm-up policy, timing boundary, sample count, and aggregation statistic;
- indicator definition, reference front/point, normalization, and numerical
  tolerance.

The repository's paper requirements and experiment scripts are the canonical
starting point for publication-oriented comparisons. Preserve their generated
manifests and the pinned environment with any reported result.

## Small reproducible run matrix

```python
from vamos import optimize

for problem in ("zdt1", "zdt2"):
    for seed in (0, 1, 2):
        result = optimize(
            problem,
            algorithm="nsgaii",
            max_evaluations=400,
            pop_size=40,
            engine="numpy",
            seed=seed,
        )
        print(problem, seed, result.F.shape, result.data["evaluations"])
```

For publication work, prefer a durable `StudySpec` so the resolved plan, task
identity, attempts, and run references remain attributable. Multiple seeds are
necessary but not sufficient for a scientific claim; choose effect sizes,
hypothesis tests, and multiplicity handling that match the experimental design.

## Optional backends

Install the `compute` extra before explicitly selecting Numba or MooCore:

```bash
python -m pip install "vamos-optimization[compute]"
```

NumPy is the reference backend. Optional acceleration depends on workload and
environment, and cross-backend bitwise equality is not promised.

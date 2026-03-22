# VAMOS vs pymoo Benchmark Report

This report uses a common seeded recipe across frameworks: matching population sizes, evaluation budgets, and operator settings for NSGA-II and MOEA/D.

- Generated: `2026-03-22T12:43:48.400207+00:00`
- VAMOS engine: `numba`
- Seeds: `42, 1337, 2024`
- Cases: `zdt1`

| Case | Algorithm | Framework | Engine | Runtime ms | HV | IGD+ | Epsilon+ | Solutions |
|------|-----------|-----------|--------|-----------:|---:|-----:|---------:|----------:|
| zdt1 | nsgaii | pymoo | pymoo | 371.66 | 0.669224 | 0.131531 | 0.145559 | 100.0 |
| zdt1 | nsgaii | vamos | numba | 1446.67 | 0.849663 | 0.014800 | 0.023509 | 100.0 |

Lower is better for runtime, IGD+, and additive epsilon. Higher is better for hypervolume.

The JSON companion includes per-seed runs and standard deviations.

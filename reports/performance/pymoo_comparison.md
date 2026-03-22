# VAMOS vs pymoo Benchmark Report

This report uses a common seeded recipe across frameworks: matching population sizes, evaluation budgets, and operator settings for NSGA-II and MOEA/D.

- Generated: `2026-03-21T22:09:32.686686+00:00`
- VAMOS engine: `numba`
- Seeds: `42`
- Cases: `dtlz2, wfg1, zdt1`

| Case | Algorithm | Framework | Engine | Runtime ms | HV | IGD+ | Epsilon+ | Solutions |
|------|-----------|-----------|--------|-----------:|---:|-----:|---------:|----------:|
| dtlz2 | moead | pymoo | pymoo | 5646.13 | 0.764312 | 0.031801 | 0.103592 | 90.0 |
| dtlz2 | moead | vamos | numba | 1434.58 | 0.774899 | 0.026413 | 0.075719 | 89.0 |
| dtlz2 | nsgaii | pymoo | pymoo | 329.02 | 0.741628 | 0.034675 | 0.137692 | 91.0 |
| dtlz2 | nsgaii | vamos | numba | 654.04 | 0.730266 | 0.040012 | 0.137198 | 91.0 |
| wfg1 | moead | pymoo | pymoo | 6309.11 | 2.722078 | 1.792168 | 2.058036 | 89.0 |
| wfg1 | moead | vamos | numba | 2418.19 | 14.233526 | 1.427410 | 1.482077 | 89.0 |
| wfg1 | nsgaii | pymoo | pymoo | 335.95 | 11.544258 | 1.496631 | 1.674751 | 91.0 |
| wfg1 | nsgaii | vamos | numba | 115.73 | 13.828815 | 1.436473 | 1.548270 | 91.0 |
| zdt1 | moead | pymoo | pymoo | 5391.20 | 0.000000 | 0.934334 | 1.171926 | 47.0 |
| zdt1 | moead | vamos | numba | 1255.03 | 0.762810 | 0.061269 | 0.186420 | 61.0 |
| zdt1 | nsgaii | pymoo | pymoo | 354.99 | 0.655074 | 0.140013 | 0.157273 | 100.0 |
| zdt1 | nsgaii | vamos | numba | 1024.64 | 0.842211 | 0.019431 | 0.027056 | 100.0 |

Lower is better for runtime, IGD+, and additive epsilon. Higher is better for hypervolume.

The JSON companion includes per-seed runs and standard deviations.

# VAMOS vs pymoo Benchmark Report

This report uses a common seeded recipe across frameworks: matching population sizes, evaluation budgets, and operator settings for NSGA-II and MOEA/D.

- Generated: `2026-03-21T22:06:44.243512+00:00`
- VAMOS engine: `numpy`
- Seeds: `42`
- Cases: `zdt1`

| Case | Algorithm | Framework | Engine | Runtime ms | HV | IGD+ | Epsilon+ | Solutions |
|------|-----------|-----------|--------|-----------:|---:|-----:|---------:|----------:|
| zdt1 | nsgaii | pymoo | pymoo | 36.86 | 0.000000 | 1.280462 | 1.570225 | 16.0 |
| zdt1 | nsgaii | vamos | numpy | 13.02 | 0.000000 | 1.416843 | 1.801677 | 9.0 |

Lower is better for runtime, IGD+, and additive epsilon. Higher is better for hypervolume.

The JSON companion includes per-seed runs and standard deviations.

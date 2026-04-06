# Farthest + GCES 3-objective ZCAT difficulty robustness campaign

This report extends the canonical 3-objective farthest/GCES survival-only campaign with official ZCAT difficulty controls. All algorithms keep the same NSGA-II host, mating, variation, and non-dominated sorting semantics. Only split-front environmental selection differs across algorithms.

## Settings

- Problems: zcat1
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvref-farthest, nsga2-hvfarthest, gces-noGeo, gces
- Seeds: [0]
- Engine: numpy
- Population size: 20
- Max evaluations: 200
- Decision variables: 30
- Objectives: 3
- Worker count: 18
- Total expected runs: 36

## Difficulty Cells

| Config | Phase | Level | Bias | Imbalance |
| --- | --- | --- | --- | --- |
| L1 | A | 1 | False | False |
| L3 | A | 3 | False | False |
| L6 | A | 6 | False | False |
| L6+B | B | 6 | True | False |
| L6+I | B | 6 | False | True |
| L6+B+I | B | 6 | True | True |

## Global Median-of-Medians

### Hypervolume

| Config | nsgaii | nsga2-farthest | nsga2-hvref-farthest | nsga2-hvfarthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+B | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+I | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+B+I | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### IGD+

| Config | nsgaii | nsga2-farthest | nsga2-hvref-farthest | nsga2-hvfarthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 3.016072 | 3.091931 | 3.457300 | 4.291180 | 3.125529 | 3.334075 |
| L3 | 15.596530 | 13.662393 | 17.950381 | 15.524854 | 20.060522 | 17.957510 |
| L6 | 37.740569 | 40.829887 | 40.634131 | 40.330652 | 40.340623 | 40.676655 |
| L6+B | 31.661283 | 31.663027 | 32.778338 | 30.622872 | 31.373405 | 30.767660 |
| L6+I | 18.208175 | 17.768700 | 16.649956 | 18.192360 | 20.255921 | 18.781521 |
| L6+B+I | 76.053081 | 70.730775 | 73.899477 | 71.806043 | 75.234930 | 71.043519 |

### Runtime (seconds)

| Config | nsgaii | nsga2-farthest | nsga2-hvref-farthest | nsga2-hvfarthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 0.027 | 1.247 | 1.327 | 1.298 | 1.199 | 1.219 |
| L3 | 0.029 | 1.129 | 1.261 | 1.172 | 1.176 | 1.127 |
| L6 | 0.043 | 1.127 | 1.251 | 1.216 | 1.190 | 1.186 |
| L6+B | 0.053 | 1.165 | 1.244 | 1.133 | 0.079 | 0.095 |
| L6+I | 0.054 | 0.051 | 0.130 | 0.112 | 0.065 | 0.072 |
| L6+B+I | 0.047 | 0.065 | 0.147 | 0.126 | 0.055 | 0.060 |

## Global Pairwise View Against nsgaii

| Config | Algorithm | HV delta | IGD+ improvement | Runtime ratio |
| --- | --- | --- | --- | --- |
| L1 | nsga2-farthest | 0.000000 | -0.075859 | 45.465 |
| L1 | nsga2-hvref-farthest | 0.000000 | -0.441228 | 48.376 |
| L1 | nsga2-hvfarthest | 0.000000 | -1.275108 | 47.333 |
| L1 | gces-noGeo | 0.000000 | -0.109456 | 43.711 |
| L1 | gces | 0.000000 | -0.318002 | 44.451 |
| L3 | nsga2-farthest | 0.000000 | 1.934137 | 39.136 |
| L3 | nsga2-hvref-farthest | 0.000000 | -2.353851 | 43.699 |
| L3 | nsga2-hvfarthest | 0.000000 | 0.071676 | 40.607 |
| L3 | gces-noGeo | 0.000000 | -4.463993 | 40.754 |
| L3 | gces | 0.000000 | -2.360980 | 39.062 |
| L6 | nsga2-farthest | 0.000000 | -3.089318 | 26.077 |
| L6 | nsga2-hvref-farthest | 0.000000 | -2.893561 | 28.935 |
| L6 | nsga2-hvfarthest | 0.000000 | -2.590083 | 28.124 |
| L6 | gces-noGeo | 0.000000 | -2.600053 | 27.527 |
| L6 | gces | 0.000000 | -2.936086 | 27.441 |
| L6+B | nsga2-farthest | 0.000000 | -0.001744 | 21.905 |
| L6+B | nsga2-hvref-farthest | 0.000000 | -1.117055 | 23.391 |
| L6+B | nsga2-hvfarthest | 0.000000 | 1.038411 | 21.301 |
| L6+B | gces-noGeo | 0.000000 | 0.287878 | 1.491 |
| L6+B | gces | 0.000000 | 0.893623 | 1.790 |
| L6+I | nsga2-farthest | 0.000000 | 0.439474 | 0.951 |
| L6+I | nsga2-hvref-farthest | 0.000000 | 1.558219 | 2.435 |
| L6+I | nsga2-hvfarthest | 0.000000 | 0.015815 | 2.094 |
| L6+I | gces-noGeo | 0.000000 | -2.047746 | 1.215 |
| L6+I | gces | 0.000000 | -0.573347 | 1.338 |
| L6+B+I | nsga2-farthest | 0.000000 | 5.322306 | 1.368 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | 2.153604 | 3.097 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | 4.247037 | 2.664 |
| L6+B+I | gces-noGeo | 0.000000 | 0.818151 | 1.151 |
| L6+B+I | gces | 0.000000 | 5.009562 | 1.274 |

## Advantage Retention Relative to L1

| Config | Algorithm | HV advantage | HV retention | IGD+ advantage | IGD+ retention |
| --- | --- | --- | --- | --- | --- |
| L1 | nsga2-farthest | 0.000000 | - | -0.075859 | 1.000 |
| L3 | nsga2-farthest | 0.000000 | - | 1.934137 | -25.496 |
| L6 | nsga2-farthest | 0.000000 | - | -3.089318 | 40.724 |
| L6+B | nsga2-farthest | 0.000000 | - | -0.001744 | 0.023 |
| L6+I | nsga2-farthest | 0.000000 | - | 0.439474 | -5.793 |
| L6+B+I | nsga2-farthest | 0.000000 | - | 5.322306 | -70.160 |
| L1 | nsga2-hvref-farthest | 0.000000 | - | -0.441228 | 1.000 |
| L3 | nsga2-hvref-farthest | 0.000000 | - | -2.353851 | 5.335 |
| L6 | nsga2-hvref-farthest | 0.000000 | - | -2.893561 | 6.558 |
| L6+B | nsga2-hvref-farthest | 0.000000 | - | -1.117055 | 2.532 |
| L6+I | nsga2-hvref-farthest | 0.000000 | - | 1.558219 | -3.532 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | - | 2.153604 | -4.881 |
| L1 | nsga2-hvfarthest | 0.000000 | - | -1.275108 | 1.000 |
| L3 | nsga2-hvfarthest | 0.000000 | - | 0.071676 | -0.056 |
| L6 | nsga2-hvfarthest | 0.000000 | - | -2.590083 | 2.031 |
| L6+B | nsga2-hvfarthest | 0.000000 | - | 1.038411 | -0.814 |
| L6+I | nsga2-hvfarthest | 0.000000 | - | 0.015815 | -0.012 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | - | 4.247037 | -3.331 |
| L1 | gces-noGeo | 0.000000 | - | -0.109456 | 1.000 |
| L3 | gces-noGeo | 0.000000 | - | -4.463993 | 40.783 |
| L6 | gces-noGeo | 0.000000 | - | -2.600053 | 23.754 |
| L6+B | gces-noGeo | 0.000000 | - | 0.287878 | -2.630 |
| L6+I | gces-noGeo | 0.000000 | - | -2.047746 | 18.708 |
| L6+B+I | gces-noGeo | 0.000000 | - | 0.818151 | -7.475 |
| L1 | gces | 0.000000 | - | -0.318002 | 1.000 |
| L3 | gces | 0.000000 | - | -2.360980 | 7.424 |
| L6 | gces | 0.000000 | - | -2.936086 | 9.233 |
| L6+B | gces | 0.000000 | - | 0.893623 | -2.810 |
| L6+I | gces | 0.000000 | - | -0.573347 | 1.803 |
| L6+B+I | gces | 0.000000 | - | 5.009562 | -15.753 |

## Additional Effect of Bias / Imbalance at Level 6

| Config | Algorithm | HV vs L6 | IGD+ vs L6 | HV advantage shift | IGD+ advantage shift |
| --- | --- | --- | --- | --- | --- |
| L6+B | nsga2-farthest | 0.000000 | -9.166860 | 0.000000 | 3.087574 |
| L6+I | nsga2-farthest | 0.000000 | -23.061187 | 0.000000 | 3.528792 |
| L6+B+I | nsga2-farthest | 0.000000 | 29.900888 | 0.000000 | 8.411624 |
| L6+B | nsga2-hvref-farthest | 0.000000 | -7.855793 | 0.000000 | 1.776506 |
| L6+I | nsga2-hvref-farthest | 0.000000 | -23.984175 | 0.000000 | 4.451780 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | 33.265346 | 0.000000 | 5.047165 |
| L6+B | nsga2-hvfarthest | 0.000000 | -9.707781 | 0.000000 | 3.628494 |
| L6+I | nsga2-hvfarthest | 0.000000 | -22.138293 | 0.000000 | 2.605898 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | 31.475391 | 0.000000 | 6.837120 |
| L6+B | gces-noGeo | 0.000000 | -8.967218 | 0.000000 | 2.887931 |
| L6+I | gces-noGeo | 0.000000 | -20.084702 | 0.000000 | 0.552307 |
| L6+B+I | gces-noGeo | 0.000000 | 34.894307 | 0.000000 | 3.418205 |
| L6+B | gces | 0.000000 | -9.908995 | 0.000000 | 3.829709 |
| L6+I | gces | 0.000000 | -21.895134 | 0.000000 | 2.362739 |
| L6+B+I | gces | 0.000000 | 30.366864 | 0.000000 | 7.945648 |

## Best Method by Difficulty Cell

| Config | Best HV | Best IGD+ |
| --- | --- | --- |
| L1 | nsgaii | nsgaii |
| L3 | nsgaii | nsga2-farthest |
| L6 | nsgaii | nsgaii |
| L6+B | nsgaii | nsga2-hvfarthest |
| L6+I | nsgaii | nsga2-hvref-farthest |
| L6+B+I | nsgaii | nsga2-farthest |

## Notes

- `summary.csv` contains per-problem per-algorithm medians and dispersion by difficulty cell.
- `comparison.csv` contains paired Wilcoxon signed-rank tests against `nsgaii` for each problem and difficulty cell.
- `global_summary.csv` collects median-of-medians and runtime trade-offs by difficulty cell.
- `difficulty_degradation.csv` and `difficulty_structure_effects.csv` quantify how quality changes as difficulty increases.

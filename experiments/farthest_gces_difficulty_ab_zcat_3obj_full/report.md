# Farthest + GCES 3-objective ZCAT difficulty robustness campaign

This report extends the canonical 3-objective farthest/GCES survival-only campaign with official ZCAT difficulty controls. All algorithms keep the same NSGA-II host, mating, variation, and non-dominated sorting semantics. Only split-front environmental selection differs across algorithms.

## Settings

- Problems: zcat1, zcat2, zcat3, zcat4, zcat5, zcat6, zcat7, zcat8, zcat9, zcat10, zcat11, zcat12, zcat13, zcat14, zcat15, zcat16, zcat17, zcat18, zcat19, zcat20
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvref-farthest, nsga2-hvfarthest, gces-noGeo, gces
- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Engine: numpy
- Population size: 100
- Max evaluations: 25000
- Decision variables: 30
- Objectives: 3
- Worker count: 18
- Total expected runs: 15120

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
| L1 | 0.773467 | 0.848516 | 0.933304 | 0.959928 | 0.846711 | 0.852918 |
| L3 | 0.099563 | 0.355128 | 0.502436 | 0.656598 | 0.320870 | 0.323207 |
| L6 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+B | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+I | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| L6+B+I | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### IGD+

| Config | nsgaii | nsga2-farthest | nsga2-hvref-farthest | nsga2-hvfarthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 0.197011 | 0.115302 | 0.059557 | 0.041088 | 0.112466 | 0.114559 |
| L3 | 0.974020 | 0.588713 | 0.421659 | 0.315292 | 0.638662 | 0.636168 |
| L6 | 32.467523 | 32.370605 | 32.309612 | 32.258253 | 32.379199 | 32.370171 |
| L6+B | 27.396320 | 27.381816 | 27.300121 | 27.294851 | 27.381272 | 27.405719 |
| L6+I | 5.379358 | 5.296848 | 5.218024 | 5.191756 | 5.373381 | 5.360745 |
| L6+B+I | 25.201568 | 24.169772 | 23.399451 | 22.651077 | 23.990684 | 24.351162 |

### Runtime (seconds)

| Config | nsgaii | nsga2-farthest | nsga2-hvref-farthest | nsga2-hvfarthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 5.156 | 8.797 | 153.291 | 164.597 | 14.379 | 19.627 |
| L3 | 4.993 | 7.192 | 87.419 | 95.363 | 10.238 | 12.279 |
| L6 | 6.095 | 8.699 | 103.328 | 109.436 | 12.536 | 15.156 |
| L6+B | 6.715 | 10.009 | 142.793 | 150.006 | 14.783 | 19.110 |
| L6+I | 5.428 | 8.624 | 130.516 | 137.580 | 13.251 | 16.606 |
| L6+B+I | 5.019 | 6.164 | 54.046 | 53.840 | 8.005 | 9.580 |

## Global Pairwise View Against nsgaii

| Config | Algorithm | HV delta | IGD+ improvement | Runtime ratio |
| --- | --- | --- | --- | --- |
| L1 | nsga2-farthest | 0.075049 | 0.081709 | 1.706 |
| L1 | nsga2-hvref-farthest | 0.159836 | 0.137454 | 29.730 |
| L1 | nsga2-hvfarthest | 0.186461 | 0.155923 | 31.923 |
| L1 | gces-noGeo | 0.073243 | 0.084545 | 2.789 |
| L1 | gces | 0.079451 | 0.082452 | 3.807 |
| L3 | nsga2-farthest | 0.255565 | 0.385306 | 1.440 |
| L3 | nsga2-hvref-farthest | 0.402873 | 0.552361 | 17.507 |
| L3 | nsga2-hvfarthest | 0.557035 | 0.658728 | 19.098 |
| L3 | gces-noGeo | 0.221307 | 0.335358 | 2.050 |
| L3 | gces | 0.223645 | 0.337852 | 2.459 |
| L6 | nsga2-farthest | 0.000000 | 0.096919 | 1.427 |
| L6 | nsga2-hvref-farthest | 0.000000 | 0.157912 | 16.953 |
| L6 | nsga2-hvfarthest | 0.000000 | 0.209270 | 17.956 |
| L6 | gces-noGeo | 0.000000 | 0.088325 | 2.057 |
| L6 | gces | 0.000000 | 0.097352 | 2.487 |
| L6+B | nsga2-farthest | 0.000000 | 0.014505 | 1.491 |
| L6+B | nsga2-hvref-farthest | 0.000000 | 0.096200 | 21.264 |
| L6+B | nsga2-hvfarthest | 0.000000 | 0.101470 | 22.339 |
| L6+B | gces-noGeo | 0.000000 | 0.015048 | 2.201 |
| L6+B | gces | 0.000000 | -0.009399 | 2.846 |
| L6+I | nsga2-farthest | 0.000000 | 0.082511 | 1.589 |
| L6+I | nsga2-hvref-farthest | 0.000000 | 0.161334 | 24.043 |
| L6+I | nsga2-hvfarthest | 0.000000 | 0.187603 | 25.344 |
| L6+I | gces-noGeo | 0.000000 | 0.005978 | 2.441 |
| L6+I | gces | 0.000000 | 0.018614 | 3.059 |
| L6+B+I | nsga2-farthest | 0.000000 | 1.031797 | 1.228 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | 1.802118 | 10.768 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | 2.550491 | 10.727 |
| L6+B+I | gces-noGeo | 0.000000 | 1.210884 | 1.595 |
| L6+B+I | gces | 0.000000 | 0.850407 | 1.909 |

## Advantage Retention Relative to L1

| Config | Algorithm | HV advantage | HV retention | IGD+ advantage | IGD+ retention |
| --- | --- | --- | --- | --- | --- |
| L1 | nsga2-farthest | 0.075049 | 1.000 | 0.081709 | 1.000 |
| L3 | nsga2-farthest | 0.255565 | 3.405 | 0.385306 | 4.716 |
| L6 | nsga2-farthest | 0.000000 | 0.000 | 0.096919 | 1.186 |
| L6+B | nsga2-farthest | 0.000000 | 0.000 | 0.014505 | 0.178 |
| L6+I | nsga2-farthest | 0.000000 | 0.000 | 0.082511 | 1.010 |
| L6+B+I | nsga2-farthest | 0.000000 | 0.000 | 1.031797 | 12.628 |
| L1 | nsga2-hvref-farthest | 0.159836 | 1.000 | 0.137454 | 1.000 |
| L3 | nsga2-hvref-farthest | 0.402873 | 2.521 | 0.552361 | 4.019 |
| L6 | nsga2-hvref-farthest | 0.000000 | 0.000 | 0.157912 | 1.149 |
| L6+B | nsga2-hvref-farthest | 0.000000 | 0.000 | 0.096200 | 0.700 |
| L6+I | nsga2-hvref-farthest | 0.000000 | 0.000 | 0.161334 | 1.174 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | 0.000 | 1.802118 | 13.111 |
| L1 | nsga2-hvfarthest | 0.186461 | 1.000 | 0.155923 | 1.000 |
| L3 | nsga2-hvfarthest | 0.557035 | 2.987 | 0.658728 | 4.225 |
| L6 | nsga2-hvfarthest | 0.000000 | 0.000 | 0.209270 | 1.342 |
| L6+B | nsga2-hvfarthest | 0.000000 | 0.000 | 0.101470 | 0.651 |
| L6+I | nsga2-hvfarthest | 0.000000 | 0.000 | 0.187603 | 1.203 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | 0.000 | 2.550491 | 16.357 |
| L1 | gces-noGeo | 0.073243 | 1.000 | 0.084545 | 1.000 |
| L3 | gces-noGeo | 0.221307 | 3.022 | 0.335358 | 3.967 |
| L6 | gces-noGeo | 0.000000 | 0.000 | 0.088325 | 1.045 |
| L6+B | gces-noGeo | 0.000000 | 0.000 | 0.015048 | 0.178 |
| L6+I | gces-noGeo | 0.000000 | 0.000 | 0.005978 | 0.071 |
| L6+B+I | gces-noGeo | 0.000000 | 0.000 | 1.210884 | 14.322 |
| L1 | gces | 0.079451 | 1.000 | 0.082452 | 1.000 |
| L3 | gces | 0.223645 | 2.815 | 0.337852 | 4.098 |
| L6 | gces | 0.000000 | 0.000 | 0.097352 | 1.181 |
| L6+B | gces | 0.000000 | 0.000 | -0.009399 | -0.114 |
| L6+I | gces | 0.000000 | 0.000 | 0.018614 | 0.226 |
| L6+B+I | gces | 0.000000 | 0.000 | 0.850407 | 10.314 |

## Additional Effect of Bias / Imbalance at Level 6

| Config | Algorithm | HV vs L6 | IGD+ vs L6 | HV advantage shift | IGD+ advantage shift |
| --- | --- | --- | --- | --- | --- |
| L6+B | nsga2-farthest | 0.000000 | -4.988789 | 0.000000 | -0.082414 |
| L6+I | nsga2-farthest | 0.000000 | -27.073757 | 0.000000 | -0.014408 |
| L6+B+I | nsga2-farthest | 0.000000 | -8.200833 | 0.000000 | 0.934878 |
| L6+B | nsga2-hvref-farthest | 0.000000 | -5.009491 | 0.000000 | -0.061712 |
| L6+I | nsga2-hvref-farthest | 0.000000 | -27.091588 | 0.000000 | 0.003423 |
| L6+B+I | nsga2-hvref-farthest | 0.000000 | -8.910161 | 0.000000 | 1.644206 |
| L6+B | nsga2-hvfarthest | 0.000000 | -4.963402 | 0.000000 | -0.107801 |
| L6+I | nsga2-hvfarthest | 0.000000 | -27.066497 | 0.000000 | -0.021668 |
| L6+B+I | nsga2-hvfarthest | 0.000000 | -9.607175 | 0.000000 | 2.341220 |
| L6+B | gces-noGeo | 0.000000 | -4.997927 | 0.000000 | -0.073276 |
| L6+I | gces-noGeo | 0.000000 | -27.005818 | 0.000000 | -0.082347 |
| L6+B+I | gces-noGeo | 0.000000 | -8.388514 | 0.000000 | 1.122560 |
| L6+B | gces | 0.000000 | -4.964452 | 0.000000 | -0.106751 |
| L6+I | gces | 0.000000 | -27.009427 | 0.000000 | -0.078738 |
| L6+B+I | gces | 0.000000 | -8.019009 | 0.000000 | 0.753054 |

## Best Method by Difficulty Cell

| Config | Best HV | Best IGD+ |
| --- | --- | --- |
| L1 | nsga2-hvfarthest | nsga2-hvfarthest |
| L3 | nsga2-hvfarthest | nsga2-hvfarthest |
| L6 | nsgaii | nsga2-hvfarthest |
| L6+B | nsgaii | nsga2-hvfarthest |
| L6+I | nsgaii | nsga2-hvfarthest |
| L6+B+I | nsgaii | nsga2-hvfarthest |

## Notes

- `summary.csv` contains per-problem per-algorithm medians and dispersion by difficulty cell.
- `comparison.csv` contains paired Wilcoxon signed-rank tests against `nsgaii` for each problem and difficulty cell.
- `global_summary.csv` collects median-of-medians and runtime trade-offs by difficulty cell.
- `difficulty_degradation.csv` and `difficulty_structure_effects.csv` quantify how quality changes as difficulty increases.

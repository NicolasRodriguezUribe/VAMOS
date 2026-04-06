# GCES 2-objective ZCAT survival-only ablation

This report describes a survival-only ablation. All four algorithms reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the GCES variants.

## Settings

- Problems: zcat1
- Algorithms: nsgaii, gces-noComp, gces-noGeo, gces
- Seeds: [0]
- Engine: numpy
- Population size: 20
- Max evaluations: 200
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- |
| zcat1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- |
| zcat1 | 1.572876 | 1.865116 | 1.865116 | 1.865116 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | gces-noComp vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | gces-noGeo vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | gces vs gces-noComp | 0/1/0 | 0/1/0 | 0/1/0 |
| zcat1 | gces vs gces-noGeo | 0/1/0 | 0/1/0 | 0/1/0 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noComp vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noGeo vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs gces-noComp | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | gces vs nsgaii | 0.292241 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noComp vs nsgaii | 0.292241 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noGeo vs nsgaii | 0.292241 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs gces-noComp | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |

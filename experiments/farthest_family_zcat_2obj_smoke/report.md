# Farthest-family 2-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the farthest-derived selectors and the GCES-family variants.

## Settings

- Problems: zcat1
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvfarthest, nsga2-refcover-farthest, nsga2-hvref-farthest, gces-noComp, gces-noGeo, gces
- Seeds: [0]
- Engine: numpy
- Population size: 20
- Max evaluations: 200
- Decision variables: 30
- Objectives: 2
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-farthest vs nsgaii, nsga2-farthest vs gces-noGeo, nsga2-farthest vs gces, nsga2-hvfarthest vs nsgaii, nsga2-hvfarthest vs nsga2-farthest, nsga2-hvfarthest vs gces-noGeo, nsga2-hvfarthest vs gces, nsga2-refcover-farthest vs nsgaii, nsga2-refcover-farthest vs nsga2-farthest, nsga2-refcover-farthest vs gces-noGeo, nsga2-refcover-farthest vs gces, nsga2-hvref-farthest vs nsgaii, nsga2-hvref-farthest vs nsga2-farthest, nsga2-hvref-farthest vs gces-noGeo, nsga2-hvref-farthest vs gces, nsga2-hvref-farthest vs nsga2-hvfarthest, nsga2-hvref-farthest vs nsga2-refcover-farthest

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 1.572876 | 1.532625 | 1.410086 | 1.806589 | 1.410086 | 1.865116 | 1.865116 | 1.865116 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-refcover-farthest | nsga2-hvref-farthest | gces-noComp | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.026 | 0.023 | 0.046 | 0.034 | 0.045 | 0.027 | 0.032 | 0.029 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-farthest vs gces-noGeo | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-farthest vs gces | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs gces | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-refcover-farthest vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-refcover-farthest vs gces | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs gces | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/1/0 | 0/1/0 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0/1/0 | 1/0/0 | 0/1/0 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | -0.040251 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | -0.332491 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | -0.332491 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | -0.162790 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | -0.122539 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | -0.455031 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces | -0.455031 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs nsgaii | 0.233714 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs nsga2-farthest | 0.273964 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs gces-noGeo | -0.058527 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-refcover-farthest vs gces | -0.058527 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsgaii | -0.162790 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | -0.122539 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | -0.455031 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces | -0.455031 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-refcover-farthest | -0.396504 | 1/0/0 | 1.000000 | 1.000000 | no |

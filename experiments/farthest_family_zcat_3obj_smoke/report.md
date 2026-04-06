# Farthest-family 3-objective ZCAT survival-only campaign

This report describes a survival-only campaign. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the farthest-derived selectors, nsga2_sector_farthest, and the GCES-family variants.

## Settings

- Problems: zcat1
- Algorithms: nsgaii, nsga2-farthest, nsga2-hvfarthest, nsga2-hvref-farthest, nsga2-sector-farthest, gces-noGeo, gces
- Seeds: [0]
- Engine: numpy
- Population size: 20
- Max evaluations: 200
- Decision variables: 30
- Objectives: 3
- Tie tolerance: 1e-12
- Wilcoxon alpha: 0.05
- Pairwise comparisons: nsga2-farthest vs nsgaii, gces-noGeo vs nsgaii, gces vs nsgaii, nsga2-farthest vs gces-noGeo, nsga2-farthest vs gces, nsga2-hvfarthest vs nsgaii, nsga2-hvfarthest vs nsga2-farthest, nsga2-hvfarthest vs gces-noGeo, nsga2-hvfarthest vs gces, nsga2-hvref-farthest vs nsgaii, nsga2-hvref-farthest vs nsga2-farthest, nsga2-hvref-farthest vs gces-noGeo, nsga2-hvref-farthest vs gces, nsga2-hvref-farthest vs nsga2-hvfarthest, nsga2-sector-farthest vs nsgaii, nsga2-sector-farthest vs nsga2-farthest, nsga2-sector-farthest vs gces-noGeo, nsga2-sector-farthest vs gces, nsga2-sector-farthest vs nsga2-hvref-farthest

## Median Hypervolume by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Median IGD+ by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 3.016072 | 3.091931 | 4.291180 | 3.457300 | 3.429009 | 3.125529 | 3.334075 |

## Median Runtime (seconds) by Problem and Algorithm

| Problem | nsgaii | nsga2-farthest | nsga2-hvfarthest | nsga2-hvref-farthest | nsga2-sector-farthest | gces-noGeo | gces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zcat1 | 0.017 | 0.025 | 0.059 | 0.063 | 0.024 | 0.022 | 0.022 |

## Seed Win Counts

| Problem | Comparison | HV W/T/L | IGD+ W/T/L | Both-metric W/T/L |
| --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | gces-noGeo vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | gces vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-farthest vs gces-noGeo | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-farthest vs gces | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvfarthest vs gces | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs gces | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0/1/0 | 1/0/0 | 0/1/0 |
| zcat1 | nsga2-sector-farthest vs nsgaii | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-sector-farthest vs gces | 0/1/0 | 0/0/1 | 0/1/0 |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0/1/0 | 1/0/0 | 0/1/0 |

## Paired Wilcoxon Signed-Rank Tests with Holm Correction

Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.

### Hypervolume

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noGeo vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsgaii | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | 0.000000 | 0/1/0 | 1.000000 | 1.000000 | no |

### IGD+

| Problem | Comparison | Median Delta | W/T/L | p_raw | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- |
| zcat1 | nsga2-farthest vs nsgaii | 0.075859 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | gces-noGeo vs nsgaii | 0.109456 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | gces vs nsgaii | 0.318002 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces-noGeo | -0.033597 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-farthest vs gces | -0.242143 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsgaii | 1.275108 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs nsga2-farthest | 1.199249 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces-noGeo | 1.165651 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvfarthest vs gces | 0.957105 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsgaii | 0.441228 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-farthest | 0.365369 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces-noGeo | 0.331772 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs gces | 0.123226 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-hvref-farthest vs nsga2-hvfarthest | -0.833880 | 1/0/0 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsgaii | 0.412937 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-farthest | 0.337078 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces-noGeo | 0.303480 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs gces | 0.094934 | 0/0/1 | 1.000000 | 1.000000 | no |
| zcat1 | nsga2-sector-farthest vs nsga2-hvref-farthest | -0.028291 | 1/0/0 | 1.000000 | 1.000000 | no |

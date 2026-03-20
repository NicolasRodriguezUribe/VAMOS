# Archive-Family Pilot Summary

## Study Description

- Input root: `/home/nico/Documents/GitHub/VAMOS/results/archive_family_pilot_core`
- Suites used: NSGAII_archive_family_DTLZ, NSGAII_archive_family_WFG, NSGAII_archive_family_ZCAT, NSGAII_archive_family_ZDT
- Variants compared: `nsgaii_archive_off`, `nsgaii_archive_passive`, `nsgaii_archive_hybrid`
- Unique seed-counts detected across suites: 10
- Final-population metrics: `hv`, `igd_plus` when available
- Archive-subset metrics: `archive_subset_hv`, `archive_subset_igd_plus` for passive and hybrid only
- Baseline has no archive subset by design, so subset comparisons are passive vs hybrid only

Planned evaluation budgets by suite:
- `DTLZ`: planned evaluations 16000 to 30000; seeds=10
- `WFG`: planned evaluations 18000 to 32000; seeds=10
- `ZCAT`: planned evaluations 16000 to 30000; seeds=10
- `ZDT`: planned evaluations 12000 to 16000; seeds=10

## Final-Population Signals By Family/Objectives

Heuristic signal columns are signed mean effects over the available final-population metrics. Positive means the left-hand variant looked better on average for that regime; negative means worse. These are pilot-effect summaries, not significance claims.

| family | n_obj | passive_vs_off_final_signal | hybrid_vs_passive_final_signal | hybrid_vs_off_final_signal | classification |
| --- | --- | --- | --- | --- | --- |
| DTLZ | 2 | 0.0000 | -0.0005 | -0.0005 | mixed/needs more budget |
| DTLZ | 3 | 0.0000 | -0.0030 | -0.0030 | mixed/needs more budget |
| DTLZ | 5 | 0.0000 | -0.0792 | -0.0792 | hybrid not yet promising |
| WFG | 2 | 0.0000 | -0.0194 | -0.0194 | mixed/needs more budget |
| WFG | 3 | 0.0000 | -0.1572 | -0.1572 | hybrid not yet promising |
| WFG | 5 | 0.0000 | -0.2906 | -0.2906 | hybrid not yet promising |
| ZCAT | 2 | 0.0000 | -0.0147 | -0.0147 | mixed/needs more budget |
| ZCAT | 3 | 0.0000 | -0.0556 | -0.0556 | hybrid not yet promising |
| ZCAT | 5 | 0.0000 | -0.5355 | -0.5355 | hybrid not yet promising |
| ZDT | 2 | 0.0000 | 0.0009 | 0.0009 | mixed/needs more budget |

## Archive-Subset Signals

Subset signals compare `hybrid_survival` against `passive` on the exported archive subset only.

| family | n_obj | hybrid_vs_passive_subset_signal | mean_reference_fraction |
| --- | --- | --- | --- |
| DTLZ | 2 | -0.0001 | 1.0000 |
| DTLZ | 3 | -0.0012 | 1.0000 |
| DTLZ | 5 | -0.0820 | 1.0000 |
| WFG | 2 | -0.0064 | 1.0000 |
| WFG | 3 | -0.0130 | 1.0000 |
| WFG | 5 | -0.0546 | 1.0000 |
| ZCAT | 2 | -0.0015 | 1.0000 |
| ZCAT | 3 | -0.0042 | 1.0000 |
| ZCAT | 5 | -0.2318 | 1.0000 |
| ZDT | 2 | 0.0017 | 0.9984 |

## Hybrid Diagnostics

| family | n_obj | hybrid_active_runs | hybrid_runtime_fallback_runs | mean_archive_reference_generations | mean_local_only_generations | mean_reference_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| DTLZ | 2 | 30 | 0 | 215.6667 | 0.0000 | 1.0000 |
| DTLZ | 3 | 30 | 0 | 265.6667 | 0.0000 | 1.0000 |
| DTLZ | 5 | 30 | 0 | 357.3333 | 0.0000 | 1.0000 |
| WFG | 2 | 30 | 0 | 232.3333 | 0.0000 | 1.0000 |
| WFG | 3 | 30 | 0 | 282.3333 | 0.0000 | 1.0000 |
| WFG | 5 | 30 | 0 | 382.3333 | 0.0000 | 1.0000 |
| ZCAT | 2 | 30 | 0 | 207.3333 | 0.0000 | 1.0000 |
| ZCAT | 3 | 30 | 0 | 257.3333 | 0.0000 | 1.0000 |
| ZCAT | 5 | 30 | 0 | 357.3333 | 0.0000 | 1.0000 |
| ZDT | 2 | 50 | 0 | 168.7400 | 0.2600 | 0.9984 |

## Overhead Snapshot

| family | n_obj | algorithm | metric | mean | median |
| --- | --- | --- | --- | --- | --- |
| DTLZ | 2 | nsgaii_archive_hybrid | archive_size | 1034.6333 | 1052.0000 |
| DTLZ | 2 | nsgaii_archive_hybrid | archive_subset_size | 80.0000 | 80.0000 |
| DTLZ | 2 | nsgaii_archive_hybrid | time_ms | 10984.7299 | 10810.0096 |
| DTLZ | 2 | nsgaii_archive_off | archive_size | 0.0000 | 0.0000 |
| DTLZ | 2 | nsgaii_archive_off | archive_subset_size | 0.0000 | 0.0000 |
| DTLZ | 2 | nsgaii_archive_off | time_ms | 10177.3522 | 10202.1975 |
| DTLZ | 2 | nsgaii_archive_passive | archive_size | 1167.2667 | 1153.0000 |
| DTLZ | 2 | nsgaii_archive_passive | archive_subset_size | 80.0000 | 80.0000 |
| DTLZ | 2 | nsgaii_archive_passive | time_ms | 10558.1053 | 10629.5051 |
| DTLZ | 3 | nsgaii_archive_hybrid | archive_size | 1980.7000 | 1564.5000 |
| DTLZ | 3 | nsgaii_archive_hybrid | archive_subset_size | 80.0000 | 80.0000 |
| DTLZ | 3 | nsgaii_archive_hybrid | time_ms | 16979.8528 | 17724.1851 |
| DTLZ | 3 | nsgaii_archive_off | archive_size | 0.0000 | 0.0000 |
| DTLZ | 3 | nsgaii_archive_off | archive_subset_size | 0.0000 | 0.0000 |
| DTLZ | 3 | nsgaii_archive_off | time_ms | 14137.1634 | 13856.5388 |
| DTLZ | 3 | nsgaii_archive_passive | archive_size | 2848.2000 | 2210.5000 |
| DTLZ | 3 | nsgaii_archive_passive | archive_subset_size | 80.0000 | 80.0000 |
| DTLZ | 3 | nsgaii_archive_passive | time_ms | 16006.6407 | 16457.3665 |

## Heuristic Decision Summary

These bullets are meant to prioritize the next larger campaign, not to make final publication claims.

### Most Promising Regimes For `hybrid_survival`

_No rows available._

### Regimes Where Passive Already Captures Most Of The Visible Gain

_No rows available._

### Regimes Where Hybrid Was Mostly Local-Only

_No rows available._

## Interpretation Notes

- Positive final-population signals for `passive_vs_off` suggest that maintaining the archive may already help in that regime.
- Positive `hybrid_vs_passive` signals suggest that archive-aware split-front survival may be adding value beyond passive archival alone.
- Low `mean_reference_fraction` means hybrid had limited opportunity to use historical archive novelty and spent more time behaving like local-only split-front selection.
- Because the baseline has no archive subset, compare subset metrics only between passive and hybrid, and use final-population metrics for baseline vs archive-family comparisons.

Related CSV artifacts:
- `archive_family_pilot_tables.csv`: per-family/per-objective aggregate tables for final population, archive subset, and overhead.
- `archive_family_pilot_by_family.csv`: family-level comparison summaries.
- `archive_family_pilot_by_objectives.csv`: objective-count comparison summaries.
- `archive_family_pilot_diagnostics.csv`: hybrid activation and fallback diagnostics.
- `archive_family_pilot_regimes.csv`: heuristic regime ranking and classification.

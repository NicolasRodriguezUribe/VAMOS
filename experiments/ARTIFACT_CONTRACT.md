# Experiment artifact contract (VAMOS)

This repository treats experiment outputs as a *contract* so that:
- runs are reproducible and auditable
- analysis/reporting can be regenerated from stored artifacts
- paper tables/figures can be built deterministically

## Run directory layout (observed in smoke run)

Each run is stored under:

results/<campaign>/<suite>/<algorithm>/<engine>/seed_<seed>/

Expected files (minimum):
- metadata.json            (structured metadata: algorithm/problem/backend/config/metrics/etc.)
- resolved_config.json     (fully resolved config used for the run)
- FUN.csv                  (final objective values; one row per solution, columns = objectives)
- X.csv                    (final decision variables; one row per solution)
- time.txt                 (runtime info; plain text)

## Tidy outputs (analysis inputs)

Collectors must write tidy tables to:
- artifacts/tidy/

Minimum tidy table for engine studies:
- artifacts/tidy/engine_smoke.csv  (one row per run)

### engine_smoke.csv (minimum columns)

Identity / provenance:
- run_path
- campaign, suite, algorithm, engine
- seed
- git_revision (if present)
- timestamp (if present)
- vamos_version (if present)

Problem / budget:
- problem
- n_obj, n_var (if present)
- population_size
- max_evaluations

Runtime:
- runtime_seconds (parsed from time.txt when possible; else blank)

Final population summary:
- front_size (rows in FUN.csv)
- fun_ncols (objective columns)
- x_ncols (variable columns)
- obj<i>_min / obj<i>_max for i=0..m-1 (derived from FUN.csv)

Metadata passthrough:
- backend_info.* (flattened if present)
- metrics.* (flattened if present)

## Paper consumption

Figures and tables generated from tidy data must be written to:
- artifacts/plots/
- artifacts/tables/

A later sync step may copy/link those into:
- paper/manuscript/figures/
- paper/manuscript/tables/

## Stopping + Archive artifacts (HV-based)

When enabled, runs MUST emit:
See `docs/experiment/stopping_and_archive.md` for config details.

### hv_trace.csv
Path: results/<campaign>/<suite>/<algo>/<engine>/seed_<seed>/hv_trace.csv

Columns (minimum):
- evals
- hv
- hv_delta
- stop_flag
- reason

### archive_stats.csv
Path: results/<campaign>/<suite>/<algo>/<engine>/seed_<seed>/archive_stats.csv

Columns (minimum):
- evals
- archive_size
- inserted
- pruned
- prune_reason

### metadata.json additions (minimum)
- stopping: {enabled, monitor_type, params, triggered, evals_stop, reason}
- archive:  {enabled, archive_type, params, final_size, total_inserted, total_pruned}

# paper/

Manuscript + supplementary material.

This tree is a `PUBLICATION_ARCHIVE`: its pinned scripts, CSV inputs, figures,
and PDFs preserve the provenance of reported numerical results. They are not a
current VAMOS StudyManifest workflow and must not be used as a study reader or
resume authority. New campaigns use the canonical study lifecycle documented
in `docs/dev/studies.md`; archived numerical results are not rewritten merely
to adopt a newer orchestration surface.

The main manuscript is `paper/manuscript/main.tex`.

Raw analysis artifacts are generated into:
- artifacts/tidy/
- artifacts/plots/
- artifacts/tables/

See: experiments/ARTIFACT_CONTRACT.md

## Build + regenerate paper tables

Install dependencies with the pinned paper environment:
- `pip install -e .`
- `pip install -r paper/requirements-publication.txt`

The pinned file captures the exact package versions used for the maintained publication stack. Use the looser extras only for exploratory local work.

Regenerate LaTeX tables from the committed CSVs:
- Runtime + solution-quality summary tables: `python paper/04_update_paper_tables_from_csv.py`
- Variant runtime tables (NSGA-II variants, SMS-EMOA, MOEA/D): `python paper/14_update_frameworks_perf_variant_tables_from_csv.py`
- Statistical appendix tables: `python paper/05_run_statistical_tests.py`
- Accessibility proxy tables: `python paper/33_update_accessibility_tables.py`

Use `--empty` on the table-update scripts to write placeholder tables when the corresponding CSV is not available yet.

## Archived paper benchmark reproduction (expensive)

Generates `experiments/benchmark_paper.csv`:
- `python paper/01_run_paper_benchmark.py`

## Run order (recommended)

1) Benchmark (cross-framework): `python paper/01_run_paper_benchmark.py`
2) Update runtime + solution-quality tables: `python paper/04_update_paper_tables_from_csv.py`
3) Update variant runtime tables: `python paper/14_update_frameworks_perf_variant_tables_from_csv.py`
4) Compile PDF + sync sources to Overleaf (excludes `main.pdf`): `python paper/08_compile_manuscript_pdf.py`
   - Compile only (no sync): `python paper/08_compile_manuscript_pdf.py --no-sync` (or `--no-sync-overleaf`)
     - Sync requires the Overleaf git remote (default name: `overleaf`) and saved credentials (Git auth token).
5) Pull sources from Overleaf to local manuscript dir (`paper/manuscript`, excludes build artifacts): `python paper/15_sync_overleaf_to_local_sources.py`
   - Preview only: `python paper/15_sync_overleaf_to_local_sources.py --dry-run`
   - Pull requires the Overleaf git remote (default name: `overleaf`) and saved credentials (Git auth token).

Controls:
- `VAMOS_N_EVALS` (default `50000`), `VAMOS_N_SEEDS` (default `30`), `VAMOS_N_JOBS`
- `VAMOS_PAPER_FRAMEWORKS` (comma-separated: `vamos-numpy,vamos-numba,vamos-moocore,pymoo,jmetalpy,deap,platypus`)
- `VAMOS_PAPER_ALGORITHM` (`nsgaii`, `smsemoa`, `moead`, or `all`) and `VAMOS_PAPER_UPDATE_MAIN_TEX` (`0`/`1`, defaults to `1` for NSGA-II)

Reproducibility note:
- Keep the benchmark CSVs, the exact command line, and `paper/requirements-publication.txt` together with any reported paper results.

## Archived supplementary / figure refresh scripts

These scripts still feed figures or appendix artifacts used by the current manuscript build:

- Scaling: `python paper/03_run_scaling_experiment.py` then `python paper/30_plot_scaling.py`
- Convergence: `python paper/18_run_convergence_experiment.py` then `python paper/19_plot_convergence.py`
- Memory: `python paper/23_run_memory_benchmark.py` then `python paper/26_plot_memory_comparison.py`
- Runtime heatmap: `python paper/28_plot_runtime_heatmap.py`
- Forest confidence intervals: `python paper/27_plot_forest_ci.py`
- Pareto front variants: `python paper/32_run_pareto_front_variants.py`
- Accessibility appendix tables: `python paper/33_update_accessibility_tables.py`
- ZCAT runtime rows used by the current tables: `python paper/31_run_zcat_all_tables.py`

## Submission packaging (Elsevier / SwEvo)

- Highlights: `paper/highlights.txt` (also included in `paper/manuscript/main.tex` via the `highlights` environment)
- Graphical abstract: see `paper/graphical_abstract.md`

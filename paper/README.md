# paper/

Manuscript + supplementary material.

The main manuscript is `paper/manuscript/main.tex`.

Raw analysis artifacts are generated into:
- artifacts/tidy/
- artifacts/plots/
- artifacts/tables/

See: experiments/ARTIFACT_CONTRACT.md

## Build + regenerate paper tables

Install dependencies (includes benchmark frameworks used in the paper):
- `pip install -e .[all]`

Regenerate LaTeX tables from the committed CSVs:
- Runtime + solution-quality summary tables: `python paper/04_update_paper_tables_from_csv.py`
- Variant runtime tables (NSGA-II variants, SMS-EMOA, MOEA/D): `python paper/14_update_frameworks_perf_variant_tables_from_csv.py`

Use `--empty` on the table-update scripts to write placeholder tables when the corresponding CSV is not available yet.

## Re-run the paper benchmark (expensive)

Generates `experiments/benchmark_paper.csv`:
- `python paper/01_run_paper_benchmark.py`

## Run order (recommended)

1) Benchmark (cross-framework): `python paper/01_run_paper_benchmark.py`
2) Update runtime + solution-quality tables: `python paper/04_update_paper_tables_from_csv.py`
3) Update variant runtime tables: `python paper/14_update_frameworks_perf_variant_tables_from_csv.py`
4) Compile PDF + sync sources to Overleaf (excludes `main.pdf`): `python paper/08_compile_manuscript_pdf.py`
   - Compile only (no sync): `python paper/08_compile_manuscript_pdf.py --no-sync` (or `--no-sync-overleaf`)
     - Sync requires the Overleaf git remote (default name: `overleaf`) and saved credentials (Git auth token).

Controls:
- `VAMOS_N_EVALS` (default `50000`), `VAMOS_N_SEEDS` (default `30`), `VAMOS_N_JOBS`
- `VAMOS_PAPER_FRAMEWORKS` (comma-separated: `vamos-numpy,vamos-numba,vamos-moocore,pymoo,jmetalpy,deap,platypus`)
- `VAMOS_PAPER_ALGORITHM` (`nsgaii`, `smsemoa`, `moead`, or `all`) and `VAMOS_PAPER_UPDATE_MAIN_TEX` (`0`/`1`, defaults to `1` for NSGA-II)

## Submission packaging (Elsevier / SwEvo)

- Highlights: `paper/highlights.txt` (also included in `paper/manuscript/main.tex` via the `highlights` environment)
- Graphical abstract: see `paper/graphical_abstract.md`

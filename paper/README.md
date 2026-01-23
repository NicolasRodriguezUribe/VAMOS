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
- Runtime tables: `python paper/04_update_paper_tables_from_csv.py`
- Statistical tables (HV + equivalence/robustness): `python paper/05_run_statistical_tests.py`
- Ablation tables: `python paper/06_update_ablation_tables_from_csv.py`
- Anytime ablation tables: `python paper/10_update_anytime_tables_from_csv.py`
- Tuned-configuration summary: `python paper/09_update_tuned_config_from_json.py`

Use `--empty` on the table-update scripts to write placeholder tables when the corresponding CSV is not available yet.

## Re-run the paper benchmark (expensive)

Generates `experiments/benchmark_paper.csv`:
- `python paper/01_run_paper_benchmark.py`

## Run order (recommended)

1) Benchmark (cross-framework): `python paper/01_run_paper_benchmark.py`
2) Ablation (VAMOS-only): `python paper/02_run_ablation_aos_racing_tuner.py`
3) Update runtime tables: `python paper/04_update_paper_tables_from_csv.py`
4) Update statistical tables: `python paper/05_run_statistical_tests.py`
5) Update ablation tables: `python paper/06_update_ablation_tables_from_csv.py`
6) Update anytime ablation tables: `python paper/10_update_anytime_tables_from_csv.py`
7) Update tuned-config table: `python paper/09_update_tuned_config_from_json.py`
8) Compile PDF + sync sources to Overleaf (excludes `main.pdf`): `python paper/08_compile_manuscript_pdf.py`
   - Compile only (no sync): `python paper/08_compile_manuscript_pdf.py --no-sync` (or `--no-sync-overleaf`)
     - Sync requires the Overleaf git remote (default name: `overleaf`) and saved credentials (Git auth token).

Controls:
- `VAMOS_N_EVALS` (default `100000`), `VAMOS_N_SEEDS` (default `30`), `VAMOS_N_JOBS`
- `VAMOS_PAPER_FRAMEWORKS` (comma-separated: `vamos-numpy,vamos-numba,vamos-moocore,pymoo,deap,jmetalpy,platypus`)

## Submission packaging (Elsevier / SwEvo)

- Highlights: `paper/highlights.txt` (also included in `paper/manuscript/main.tex` via the `highlights` environment)
- Graphical abstract: see `paper/graphical_abstract.md`

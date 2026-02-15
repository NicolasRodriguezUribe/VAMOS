# CLI and config files

Quickstart wizard
-----------------

Run an interactive wizard that writes a config file and executes a single run:

```bash
vamos quickstart
```

If you are new to Python, start with `docs/guide/minimal-python.md`.

List available templates:

```bash
vamos quickstart --template list
```

Run a template non-interactively:

```bash
vamos quickstart --template physics_design --yes --no-plot
```

Skip optional dependency warnings:

```bash
vamos quickstart --no-preflight
```

Template keys (short list):

- `demo`: quick benchmark demo (ZDT1)
- `physics_design`: mixed-variable structural design (welded beam)
- `bio_feature_selection`: real-data feature selection (requires `examples` extra)
- `chem_hyperparam_tuning`: SVM hyperparameter tuning (requires `examples` extra)

The config is saved under `results/quickstart/` and can be re-run with `vamos --config <path>`.

Results helpers
---------------

Summarize recent runs:

```bash
vamos summarize --results results
```

Show only the latest run:

```bash
vamos summarize --latest
```

Open the latest run folder:

```bash
vamos open-results --open
```

Main runner
-----------

Use `python -m vamos.experiment.cli.main` (or `vamos`) for single runs and problem sets.

Quick walkthroughs
------------------

Single run (default output under `results/`):

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7
```

Steady-state NSGA-II (incremental replacement):

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --nsgaii-steady-state --nsgaii-replacement-size 2
```

Python equivalent (preferred for scripting):

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=5000, pop_size=80, seed=7)
```

Run a predefined problem set with both internal algorithms:

```bash
vamos --problem-set families --algorithm both --max-evaluations 3000
```

Compare backends on one problem:

```bash
vamos --problem zdt1 --experiment backends --max-evaluations 2000
```

Optional backends need extras: `numba`/`moocore` require `pip install -e ".[compute]"` (or `pip install "vamos[compute]"`), and `jax` requires `pip install -e ".[autodiff]"` (or `pip install "vamos[autodiff]"`). Missing backends are skipped.

JAX run (strict ranking fallback for exact Pareto fronts):

```bash
vamos --problem zdt1 --algorithm nsgaii --engine jax --max-evaluations 5000
```

Requires the optional `autodiff` extra (`pip install -e ".[autodiff]"` or `pip install "vamos[autodiff]"`).

Multiprocessing evaluation for expensive problems:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 8000 --eval-strategy multiprocessing --n-workers 4
```

Enable live visualization and save plots:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 2000 --live-viz --plot
```

Early stop when hypervolume reaches a target fraction:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 15000 --hv-threshold 0.9
```

Include external baselines (ZDT1 only):

```bash
vamos --problem zdt1 --algorithm both --include-external --external-problem-source native
```

Walkthrough: run and inspect outputs
------------------------------------

1) Run a single optimization:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7
```

2) Inspect artifacts under `results/` (default):

- `FUN.csv`: objective values (Pareto front)
- `X.csv`: decision variables (if exported)
- `metadata.json`: run configuration and timings
- `resolved_config.json`: resolved settings and inferred defaults

3) Save plots in the same folder:

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7 --plot
```

Key flags
---------

- `--algorithm`: nsgaii, moead, smsemoa, nsgaiii, spea2, ibea, smpso, both, or external baselines (pymoo_nsga2, jmetalpy_nsga2, pygmo_nsga2)
- `--engine`: numpy | numba | moocore | jax (strict ranking uses NumPy fallback; set `VAMOS_JAX_STRICT_RANKING=0` for approximate ranking)
- `--problem`: any registry key (see Problems page)
- `--problem-set`: predefined sets (e.g., `families`)
- `--validate-config`: validate `--config` and exit
- `--output-root`: directory for run artifacts (default: `results/`)
- `--no-preflight`: skip optional dependency warnings
- `--population-size`, `--offspring-population-size`
- NSGA-II steady-state:
  - `--nsgaii-steady-state` to enable incremental replacement
  - `--nsgaii-replacement-size` to set replacements per step (implies steady-state)
- `--max-evaluations`
- `--hv-threshold` and `--hv-reference-front`
- `--selection-pressure`, `--external-archive-size`
- `--eval-strategy`: serial | multiprocessing (with `--n-workers`)
- `--live-viz` with `--live-viz-interval`, `--live-viz-max-points`
- `--plot`: save Pareto front plots after runs
- Variation overrides per algorithm (examples):
  - `--nsgaii-crossover sbx --nsgaii-crossover-prob 1.0 --nsgaii-mutation pm --nsgaii-mutation-prob 1/n`
  - `--moead-crossover sbx --moead-mutation pm --moead-aggregation pbi`
  - `--smsemoa-mutation pm --nsga3-crossover sbx`

Config files (YAML/JSON)
------------------------

Use `--config path/to/spec.yaml`; CLI flags override file values.

```yaml
version: "1"
defaults:
  title: My run
  algorithm: moead
  engine: numpy
  population_size: 120
  max_evaluations: 20000
  hv_threshold: 0.8
  moead:
    crossover: {method: sbx, prob: 1.0, eta: 20}
    mutation: {method: pm, prob: "1/n", eta: 20}
problems:
  bin_knapsack:
    algorithm: nsgaii
    n_var: 30
    population_size: 150
    nsgaii:
      crossover: {method: uniform}
      mutation: {method: bitflip, prob: "1/n"}
```

Validate a config without running:

```bash
vamos --config configs/experiment.yaml --validate-config
```

Run a config with a CLI override:

```bash
vamos --config configs/experiment.yaml --algorithm smsemoa --max-evaluations 10000
```

Other subcommands
-----------------

All tools are accessed via `vamos <subcommand>`. Run `vamos help` for the full list.

- Self-check: `vamos check`
- Benchmarking: `vamos bench --list` and `vamos bench ZDT_small --algorithms nsgaii moead --output report/`
- Tuning: `vamos tune --instances zdt1,zdt2,zdt3 --algorithm nsgaii --backend optuna --backend-fallback random --split-strategy suite_stratified --budget 5000 --tune-budget 200 --n-jobs -1`
- Ablation plans: `vamos ablation --config configs/ablation.yaml`
- Profiling: `vamos profile --problem zdt1 --engines numpy,numba --budget 2000 --output report/profile.csv`
- Problem zoo: `vamos zoo list`, `vamos zoo info zdt1`, `vamos zoo run zdt1 --algorithm nsgaii --budget 3000`
- Studio (interactive, needs `studio` extra): `vamos studio --study-dir results`

Tuning quick notes (`vamos tune`)
---------------------------------

Use this guide for quick usage. For the complete, maintained `tune` reference
(all backends, split/fallback behavior, finisher/validation/test, and artifact
contracts), see:

- `docs/topics/tuning.md`

Recommended robust invocation:

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend optuna \
  --backend-fallback random \
  --split-strategy suite_stratified \
  --budget 5000 \
  --tune-budget 200 \
  --n-jobs -1
```

Ablation config example
-----------------------

```yaml
algorithm: nsgaii
engine: numpy
output_root: results/ablation_demo
default_max_evals: 2000
problems: [zdt1]
seeds: [1, 2, 3]
base_config:
  population_size: 60
  offspring_population_size: 60
variants:
  - name: baseline
  - name: aos
    nsgaii_variation:
      adaptive_operator_selection:
        enabled: true
summary_dir: results/ablation_demo/summary
```

The CLI writes a summary CSV by default to `<output_root>/summary/ablation_metrics.csv` (override with `summary_path` or `summary_dir`).

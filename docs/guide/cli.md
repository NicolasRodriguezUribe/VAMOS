# CLI and config files

Main runner
-----------

Use `python -m vamos.experiment.cli.main` (or `vamos`) for single runs and problem sets.

Quick walkthroughs
------------------

Single run (default output under `results/`):

```bash
vamos --problem zdt1 --algorithm nsgaii --max-evaluations 5000 --population-size 80 --seed 7
```

Python equivalent (preferred for scripting):

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", budget=5000, pop_size=80, seed=7)
```

Run a predefined problem set with both internal algorithms:

```bash
vamos --problem-set families --algorithm both --max-evaluations 3000
```

Compare backends on one problem:

```bash
vamos --problem zdt1 --experiment backends --max-evaluations 2000
```

JAX run (strict ranking fallback for exact Pareto fronts):

```bash
vamos --problem zdt1 --algorithm nsgaii --engine jax --max-evaluations 5000
```

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

Key flags
---------

- `--algorithm`: nsgaii, moead, smsemoa, nsgaiii, spea2, ibea, smpso, both, or external baselines (pymoo_nsga2, jmetalpy_nsga2, pygmo_nsga2)
- `--engine`: numpy | numba | moocore | jax (strict ranking uses NumPy fallback; set `VAMOS_JAX_STRICT_RANKING=0` for approximate ranking)
- `--problem`: any registry key (see Problems page)
- `--problem-set`: predefined sets (e.g., `families`)
- `--validate-config`: validate `--config` and exit
- `--output-root`: directory for run artifacts (default: `results/`)
- `--population-size`, `--offspring-population-size`
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

Other CLIs
----------

- Self-check: `vamos-self-check`
- Benchmarking: `vamos-benchmark --list` and `vamos-benchmark ZDT_small --algorithms nsgaii moead --output report/`
- Tuning: `vamos-tune --problem zdt1 --algorithm nsgaii --budget 5000 --tune-budget 20000 --n-jobs 4`
- Profiling: `vamos-profile --problem zdt1 --engines numpy,numba --budget 2000 --output report/profile.csv`
- Problem zoo: `vamos-zoo list`, `vamos-zoo info zdt1`, `vamos-zoo run zdt1 --algorithm nsgaii --budget 3000`
- Studio (interactive, needs `studio` extra): `vamos-studio --study-dir results`

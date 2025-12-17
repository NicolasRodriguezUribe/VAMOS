# CLI and config files

Main runner
-----------

`python -m vamos.experiment.cli.main` (or `vamos`) key flags:

- `--algorithm`: nsgaii, moead, smsemoa, nsga3, spea2, ibea, smpso, both, or external baselines (pymoo_nsga2, jmetalpy_nsga2, pygmo_nsga2)
- `--engine`: numpy | numba | moocore
- `--problem`: any registry key (see Problems page)
- `--problem-set`: predefined sets (e.g., `families`)
- `--population-size`, `--offspring-population-size`
- `--max-evaluations`
- `--hv-threshold` and `--hv-reference-front`
- `--selection-pressure`, `--external-archive-size`
- `--eval-backend`: serial | multiprocessing (with `--n-workers`)
- `--live-viz` with `--live-viz-interval`, `--live-viz-max-points`
- Variation overrides per algorithm (examples):
  - `--nsgaii-crossover sbx --nsgaii-crossover-prob 0.9 --nsgaii-mutation pm --nsgaii-mutation-prob 1/n`
  - `--moead-crossover sbx --moead-mutation pm --moead-aggregation tchebycheff`
  - `--smsemoa-mutation pm --nsga3-crossover sbx`

Config files (YAML/JSON)
------------------------

Use `--config path/to/spec.yaml`; CLI flags override file values.

```yaml
defaults:
  title: My run
  algorithm: moead
  engine: numpy
  population_size: 120
  max_evaluations: 20000
  hv_threshold: 0.8
  moead:
    crossover: {method: sbx, prob: 0.9, eta: 20}
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

Other CLIs
----------

- Benchmarking: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/`
- Study runner: `python -m vamos.experiment.study.runner --help`
- Studio (interactive, needs `studio` extra): `vamos-studio --study-dir results`

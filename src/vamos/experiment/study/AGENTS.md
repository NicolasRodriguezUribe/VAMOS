# Study Module

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).


This directory contains batch experiment orchestration for VAMOS.

## Purpose

Run systematic studies: problem × algorithm × seed combinations.

## StudyRunner

```python
from vamos.experiment.study.runner import StudyRunner, StudyTask

tasks = [
    StudyTask(problem="zdt1", algorithm="nsgaii", seed=1, max_evals=25000),
    StudyTask(problem="zdt1", algorithm="nsgaii", seed=2, max_evals=25000),
    StudyTask(problem="zdt1", algorithm="moead", seed=1, max_evals=25000),
    # ... more combinations
]

runner = StudyRunner(tasks, output_dir="results/my_study", engine="numpy")
runner.run_all()
```

## Study Definition via Config

```yaml
# study.yaml
defaults:
  engine: numpy
  population_size: 100
  max_evaluations: 25000

problems:
  - zdt1
  - zdt2
  - dtlz2

algorithms:
  - nsgaii
  - moead
  - spea2

seeds: [1, 2, 3, 4, 5]
output_dir: results/benchmark_study
```

## Output Structure

```
results/my_study/
├── zdt1/
│   ├── nsgaii_seed1/
│   │   ├── front.csv
│   │   └── metadata.json
│   ├── nsgaii_seed2/
│   └── moead_seed1/
├── zdt2/
└── summary.csv
```

## Key Files

| File | Purpose |
|------|---------|
| `runner.py` | StudyRunner, StudyTask classes |
| `config.py` | Study configuration parsing |
| `aggregator.py` | Results collection utilities |

## Analysis After Study

```python
from vamos.ux.analysis.loader import load_study_results
from vamos.ux.analysis.stats import friedman_test

df = load_study_results("results/my_study")
friedman_test(df, metric="hv")
```

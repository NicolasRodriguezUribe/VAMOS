# Run Benchmark Study

Use this prompt when setting up a reproducible benchmark study across problems, algorithms, and seeds.

## Task
Run a benchmark comparing `{ALGORITHMS}` on `{PROBLEMS}` with `{N_SEEDS}` independent runs.

## Steps

1. **Create study config** (`studies/{study_name}.yaml`):
   ```yaml
   defaults:
     engine: numpy
     population_size: 100
     max_evaluations: 25000
   
   problems:
     zdt1:
       n_var: 30
     zdt2:
       n_var: 30
     dtlz2:
       n_var: 12
       n_obj: 3
   
   algorithms:
     - nsgaii
     - moead
     - spea2
   
   seeds: [1, 2, 3, 4, 5]
   
   output_dir: results/{study_name}
   ```

2. **Run via CLI**
   ```bash
   python -m vamos.experiment.cli.main --config studies/{study_name}.yaml
   ```

3. **Or use StudyRunner programmatically**
   ```python
   from vamos.experiment.study.runner import StudyRunner, StudyTask
   
   tasks = [
       StudyTask(problem="zdt1", algorithm="nsgaii", seed=s, max_evals=25000)
       for s in range(1, 6)
   ]
   runner = StudyRunner(tasks, output_dir="results/my_study")
   runner.run_all()
   ```

4. **Analyze results**
   ```python
   from vamos.analysis.stats import friedman_test, plot_critical_distance
   from vamos.analysis.loader import load_study_results
   
   df = load_study_results("results/{study_name}")
   friedman_test(df, metric="hv")
   plot_critical_distance(df, metric="hv")
   ```

## Output Structure
```
results/{study_name}/
├── zdt1/
│   ├── nsgaii_seed1/
│   │   ├── front.csv
│   │   └── metadata.json
│   ├── nsgaii_seed2/
│   └── ...
├── zdt2/
└── summary.csv
```

## Checklist
- [ ] Config file with all problems/algorithms/seeds
- [ ] Output directory specified
- [ ] Seeds for reproducibility
- [ ] Reference fronts available for HV computation
- [ ] Post-hoc statistical analysis planned

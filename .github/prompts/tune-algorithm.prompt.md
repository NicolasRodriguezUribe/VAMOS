# Tune Algorithm Hyperparameters

Use this prompt when setting up AutoNSGA-II or other meta-optimization for algorithm tuning.

## Task
Tune `{ALGORITHM}` hyperparameters on `{PROBLEMS}` to optimize `{INDICATOR}` (e.g., hypervolume).

## Steps

1. **Define config space**
   ```python
   from vamos.tuning import AlgorithmConfigSpace
   
   space = AlgorithmConfigSpace()
   
   # Population parameters
   space.add_int("pop_size", 50, 200)
   space.add_int("offspring_size", 10, 100)
   
   # Crossover
   space.add_categorical("crossover", ["sbx", "blx_alpha"])
   space.add_float("crossover_prob", 0.7, 1.0)
   space.add_float("crossover_eta", 5.0, 30.0)  # for SBX
   
   # Mutation
   space.add_float("mutation_prob_factor", 0.5, 2.0)  # multiplier for 1/n
   space.add_float("mutation_eta", 5.0, 30.0)
   
   # Selection
   space.add_int("tournament_size", 2, 10)
   ```

2. **Or use template**
   ```python
   space = AlgorithmConfigSpace.from_template("nsgaii", "default")
   ```

3. **Setup tuner**
   ```python
   from vamos.tuning import NSGAIITuner
   from vamos.problem.registry import make_problem_selection
   
   problems = [
       make_problem_selection("zdt1", n_var=30).instantiate(),
       make_problem_selection("zdt2", n_var=30).instantiate(),
   ]
   
   tuner = NSGAIITuner(
       config_space=space,
       problems=problems,
       ref_fronts=[None, None],  # or load reference fronts
       indicators=["hv"],
       max_evals_per_problem=10000,
       n_runs_per_problem=3,
       engine="numpy",
       meta_population_size=20,
       meta_max_evals=100,
       seed=42,
   )
   ```

4. **Run and extract best**
   ```python
   X_meta, F_meta, configs, diagnostics = tuner.optimize()
   
   best_idx = int(np.argmin(F_meta[:, 0]))
   best_config = configs[best_idx]
   print("Best config:", best_config)
   ```

## Checklist
- [ ] Config space covers relevant hyperparameters
- [ ] Multiple problems for generalization
- [ ] Appropriate inner budget (max_evals_per_problem)
- [ ] Multiple runs per config (n_runs_per_problem) for robustness
- [ ] Seed set for reproducibility
- [ ] Reference fronts for accurate HV (optional)

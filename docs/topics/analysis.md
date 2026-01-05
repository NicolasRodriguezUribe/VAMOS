# Analysis and Visualization

VAMOS provides a suite of tools for responding to the "Now what?" question after optimization: statistical testing, result aggregation, and publication-ready plotting.

## Statistics

The `vamos.ux.analysis.stats` module includes non-parametric tests suitable for evolutionary algorithm comparison.

```python
from vamos.ux.analysis.stats import friedman_test, pairwise_wilcoxon, perform_statistical_tests

# metric_dict maps {algorithm_name: [list_of_scores_across_seeds]}
results = {
    "NSGA-II": [0.85, 0.86, 0.84],
    "MOEA/D": [0.82, 0.81, 0.83],
    "SPEA2": [0.85, 0.85, 0.86]
}

# Run Friedman rank sum test
f_stat, p_value, rankings, avg_ranks = friedman_test(results)

# Run post-hoc Wilcoxon signed-rank tests with Holm correction
p_values, reject = pairwise_wilcoxon(results)
```

## Landscape Analysis

Understanding problem difficulty often requires analyzing the fitness landscape.

```python
from vamos.analysis.landscape import perform_random_walk, compute_autocorrelation

# Perform a random walk on the problem instance
f_values = perform_random_walk(problem, n_steps=1000)

# Compute autocorrelation (measure of rugosity)
corr = compute_autocorrelation(f_values, lag=1)
print(f"Lag-1 Autocorrelation: {corr:.2f}")
# Closer to 1.0 = smooth; Closer to 0.0 (or negative) = rugged
```

## Visualization

Helper functions in `vamos.ux.analysis.plotting` (and `vamos.plotting`) simplify common tasks.

### Critical Distance Plots
Visualize statistical significance groups following the Friedman/Nemenyi post-hoc style.

```python
from vamos.ux.analysis.stats import plot_critical_distance

plot_critical_distance(
    avg_ranks, 
    num_datasets=10,  # number of problems/seeds
    filename="cd_plot.tex"
)
```

### Pareto Fronts

```python
from vamos.plotting import plot_pareto_front_2d

# F is an (N, 2) array of objectives
plot_pareto_front_2d(F, title="ZDT1 Result", filename="front.png")
```

## MCDM (Multi-Criteria Decision Making)

Select specific solutions from a Pareto front using `vamos.mcdm`.

- **Weighted Sum**: `weighted_sum_scores(F, weights)`
- **TOPSIS**: `topsis_scores(F, weights)`
- **Knee Point**: Find the "knee" of the curve.

```python
from vamos.mcdm import weighted_sum_selection

best_idx = weighted_sum_selection(F, weights=[0.5, 0.5])
best_solution = X[best_idx]
```

# Analysis and Visualization

VAMOS provides a suite of tools for responding to the "Now what?" question after optimization: statistical testing, result aggregation, and publication-ready plotting.

## Statistics

The `vamos.ux.api` facade exposes non-parametric tests suitable for evolutionary algorithm comparison.

```python
from vamos.ux.api import friedman_test, pairwise_wilcoxon

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

For landscape analysis workflows (random walks, autocorrelation, ruggedness), see:

- `notebooks/2_advanced/25_landscape_analysis.ipynb`

## Visualization

Helper functions in `vamos.ux.api` simplify common tasks.

### Critical Distance Plots
Visualize statistical significance groups following the Friedman/Nemenyi post-hoc style.

```python
from vamos.ux.api import plot_critical_distance

plot_critical_distance(
    avg_ranks, 
    num_datasets=10,  # number of problems/seeds
    filename="cd_plot.tex"
)
```

### Pareto Fronts

```python
from vamos.ux.api import plot_pareto_front_2d

# F is an (N, 2) array of objectives
plot_pareto_front_2d(F, title="ZDT1 Result", filename="front.png")
```

## MCDM (Multi-Criteria Decision Making)

Select specific solutions from a Pareto front using `vamos.ux.api`.

- **Weighted Sum**: `weighted_sum_scores(F, weights)`
- **TOPSIS**: `topsis_scores(F, weights)`
- **Knee Point**: Find the "knee" of the curve.

```python
from vamos.ux.api import weighted_sum_scores

scores = weighted_sum_scores(F, weights=[0.5, 0.5]).scores
best_idx = int(scores.argmin())
best_solution = X[best_idx]
```

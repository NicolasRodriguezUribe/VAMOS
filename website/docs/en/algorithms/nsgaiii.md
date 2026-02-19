# NSGA-III

Non-dominated Sorting Genetic Algorithm III. Extends NSGA-II for many-objective problems by replacing crowding distance with a reference-point-based diversity mechanism. Solutions are selected based on their distance to structured reference directions on the unit simplex, maintaining coverage across the entire objective space.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Dominance ranking + reference point association |
| Objectives | 3–10+ (designed for ≥ 4) |
| Ask/Tell | Yes |
| Best use case | Many-objective problems where structured diversity is needed |

---

## Minimal example

```python
from vamos import optimize

# 3-objective DTLZ2
result = optimize("dtlz2", algorithm="nsgaiii", max_evaluations=30000, seed=42)

print(result.F.shape)  # (91, 3) — size depends on reference points
```

Custom problem with 5 objectives:

```python
from vamos import make_problem, optimize
import numpy as np

def dtlz2_5obj(x):
    g = np.sum((x[2:] - 0.5) ** 2)
    cos = [np.cos(xi * np.pi / 2) for xi in x[:2]]
    sin = [np.sin(xi * np.pi / 2) for xi in x[:2]]
    f = [(1 + g) * cos[0] * cos[1],
         (1 + g) * cos[0] * sin[1],
         (1 + g) * sin[0],
         (1 + g) * (1 - x[0]),
         (1 + g) * x[1]]
    return f

problem = make_problem(dtlz2_5obj, n_var=7, n_obj=5, xl=0.0, xu=1.0)
result = optimize(problem, algorithm="nsgaiii", max_evaluations=50000, seed=42)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | inferred | Inferred from `ref_dirs` if not set. |
| `ref_dirs` | auto | Reference directions. Auto-generated via Das–Dennis decomposition if not provided. Override with `n_partitions` in `algorithm_kwargs`. |
| `n_partitions` | auto | Number of Das–Dennis partitions per objective. Controls resolution of reference points. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 30 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |

```python
result = optimize(
    "dtlz2", algorithm="nsgaiii", max_evaluations=50000, seed=0,
    algorithm_kwargs={"n_partitions": 12},
)
```

---

## When to use

**Use NSGA-III when:**

- You have 4 or more objectives.
- You want structured diversity coverage over the objective space.
- The Pareto front is approximately simplex-shaped.

**Consider alternatives when:**

- You have only 2–3 objectives — [NSGA-II](nsgaii.md) is simpler and often faster.
- The Pareto front has an irregular or degenerate geometry — try [AGE-MOEA](agemoea.md) or [RVEA](rvea.md).
- You want adaptive reference vectors — try [RVEA](rvea.md).

---

## Reference

Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601. <https://doi.org/10.1109/TEVC.2013.2281535>

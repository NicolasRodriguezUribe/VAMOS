# RVEA

Reference Vector Guided Evolutionary Algorithm. Guides selection using a set of reference vectors uniformly distributed on the unit hypersphere. The angle between each candidate solution and the nearest reference vector controls selection pressure. Reference vectors are periodically adapted toward the current population's Pareto front to handle irregular geometries.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Reference-vector-guided EA with adaptive vectors |
| Objectives | 5–15 (designed for many-objective) |
| Ask/Tell | Yes |
| Best use case | Many-objective problems where reference vectors can be tuned to the front |

---

## Minimal example

```python
from vamos import optimize

# 8-objective DTLZ2
result = optimize(
    "dtlz2",
    algorithm="rvea",
    max_evaluations=100000,
    seed=42,
    algorithm_kwargs={"n_obj": 8},
)

print(result.F.shape)
```

With explicit reference vector configuration:

```python
result = optimize(
    "dtlz3",
    algorithm="rvea",
    max_evaluations=80000,
    seed=0,
    algorithm_kwargs={"n_partitions": 6, "fr": 0.1, "alpha": 2.0},
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | inferred | Inferred from reference vector count. |
| `n_partitions` | auto | Das–Dennis partition parameter; controls reference vector count. |
| `alpha` | 2.0 | Angle-penalized distance parameter. Controls convergence vs. diversity balance. |
| `fr` | 0.1 | Reference vector adaptation frequency as a fraction of `max_evaluations`. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |

---

## When to use

**Use RVEA when:**

- You have 5 or more objectives.
- You want reference vectors that adapt to the true Pareto front geometry during optimization.
- You need fine-grained control over the reference direction distribution.

**Consider alternatives when:**

- You have 2–3 objectives — [NSGA-II](nsgaii.md) is faster and requires less configuration.
- The Pareto front is highly irregular and reference vectors cannot adapt fast enough — try [AGE-MOEA](agemoea.md).
- You want a reference-point-free approach entirely — try [AGE-MOEA](agemoea.md).

---

## Reference

Cheng, R., Jin, Y., Olhofer, M., & Sendhoff, B. (2016). A reference vector guided evolutionary algorithm for many-objective optimization. *IEEE Transactions on Evolutionary Computation*, 20(5), 773–791. <https://doi.org/10.1109/TEVC.2016.2519378>

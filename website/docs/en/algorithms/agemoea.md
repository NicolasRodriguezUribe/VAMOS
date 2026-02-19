# AGE-MOEA

Adaptive Geometry Estimation-based Multi-Objective Evolutionary Algorithm. Estimates the local geometry of the Pareto front at each generation and adjusts the diversity preservation metric accordingly. Requires no reference points — the algorithm infers the front shape from the current population. Handles irregular, degenerate, and mixed-geometry fronts well.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Adaptive geometry estimation + dominance ranking |
| Objectives | 2–10+ |
| Ask/Tell | Yes |
| Best use case | Problems with unknown or irregular Pareto front geometry |

---

## Minimal example

```python
from vamos import optimize

result = optimize("dtlz2", algorithm="agemoea", max_evaluations=30000, seed=42)

print(result.F.shape)  # (100, 3)
```

5-objective problem — no reference point configuration needed:

```python
result = optimize(
    "dtlz4",
    algorithm="agemoea",
    max_evaluations=50000,
    seed=42,
    algorithm_kwargs={"pop_size": 200},
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Population size. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |

---

## When to use

**Use AGE-MOEA when:**

- You don't know the shape of the Pareto front in advance.
- The front has mixed curvature — some regions convex, others concave or degenerate.
- You want many-objective capability without configuring reference points.
- You are comparing against a reference-point-free many-objective baseline.

**Consider alternatives when:**

- The front shape is known to be a unit simplex — [NSGA-III](nsgaiii.md) with matching reference points may outperform.
- You need reproducible diversity control via explicit reference structure — try [RVEA](rvea.md).
- You have only 2–3 objectives — [NSGA-II](nsgaii.md) is faster and simpler.

---

## Reference

Panichella, A. (2019). An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective optimization. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2019)*, pp. 595–603. <https://doi.org/10.1145/3321707.3321839>

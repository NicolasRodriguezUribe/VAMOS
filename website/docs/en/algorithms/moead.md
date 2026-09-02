# MOEA/D

Multi-Objective Evolutionary Algorithm Based on Decomposition. Converts the multi-objective problem into a set of scalar subproblems using weight vectors. Each subproblem is solved cooperatively by sharing information with its neighborhood. Memory-efficient — one objective evaluation per subproblem per generation.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Decomposition (Tchebycheff scalarization) |
| Objectives | 2–5 (weight vector geometry degrades on 6+) |
| Ask/Tell | Yes |
| Best use case | Problems with regular, well-structured Pareto fronts |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="moead", max_evaluations=20000, seed=42)

print(result.F.shape)  # (n_solutions, 2)
```

With an explicit public configuration:

```python
from vamos.algorithms import MOEADConfig

config = MOEADConfig.default(pop_size=100, n_var=30, n_obj=2)
result = optimize(
    "zdt2",
    algorithm="moead",
    max_evaluations=30000,
    seed=42,
    algorithm_config=config,
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Number of subproblems. Must equal the number of weight vectors. |
| `neighbor_size` | 20 | Neighborhood size per weight vector. |
| `aggregation` | `pbi(theta=5.0)` | Scalar aggregation configuration. |
| `delta` | 0.9 | Probability of sampling parents from the neighborhood. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |

---

## When to use

**Use MOEA/D when:**

- The problem has a smooth, convex, or known-geometry Pareto front.
- You want uniform front coverage with a fixed set of weight vectors.
- Memory or computational efficiency is a concern (MOEA/D generates one offspring per subproblem).

**Consider alternatives when:**

- The Pareto front is irregular, disconnected, or has extreme curvature — weight vectors may not map well.
- You have 6+ objectives — weight vector uniformity degrades in high dimensions.
- You want hypervolume-maximizing selection — try [SMS-EMOA](smsemoa.md).

---

## Reference

Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712–731. <https://doi.org/10.1109/TEVC.2007.892759>

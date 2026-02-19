# SPEA2

Strength Pareto Evolutionary Algorithm 2. Maintains an external archive of non-dominated solutions alongside the main population. Fitness combines a Pareto dominance strength value with a k-nearest-neighbor density estimate. Archive truncation preserves boundary solutions when the archive overflows.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Archive-based dominance ranking + density estimation |
| Objectives | 2–3 |
| Ask/Tell | Yes |
| Best use case | Problems requiring a curated, fixed-size archive of non-dominated solutions |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="spea2", max_evaluations=10000, seed=42)

print(result.F.shape)  # (100, 2) — returns the archive contents
```

With explicit parameters:

```python
result = optimize(
    "zdt3",
    algorithm="spea2",
    max_evaluations=20000,
    seed=42,
    algorithm_kwargs={"pop_size": 100, "archive_size": 100},
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Main population size. |
| `archive_size` | 100 | External archive capacity. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |
| `k` | auto | k-NN density estimate parameter. Default: `sqrt(pop_size + archive_size)`. |

---

## When to use

**Use SPEA2 when:**

- You need a fixed-size, curated archive of high-quality non-dominated solutions.
- The archive-based selection gives more stable convergence on your problem.
- Comparison with other archive-based methods (e.g. from the early 2000s literature) is needed.

**Consider alternatives when:**

- Budget is tight — archive operations add O(N²) distance computations per generation.
- You have 4+ objectives — try [NSGA-III](nsgaiii.md) or [AGE-MOEA](agemoea.md).
- You need the absolute best hypervolume — try [SMS-EMOA](smsemoa.md).

---

## Reference

Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the strength Pareto evolutionary algorithm. *TIK-Report 103*, ETH Zurich. <https://doi.org/10.3929/ethz-a-004284029>

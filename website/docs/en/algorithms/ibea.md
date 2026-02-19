# IBEA

Indicator-Based Evolutionary Algorithm. Uses a binary quality indicator — additive epsilon (ε⁺) or hypervolume — directly as the selection criterion. No explicit diversity mechanism is needed because the indicator already encodes both convergence and spread information.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Quality indicator (ε⁺ or hypervolume) |
| Objectives | 2–3 |
| Ask/Tell | Yes |
| Best use case | Directly optimizing a specific quality indicator |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="ibea", max_evaluations=10000, seed=42)

print(result.F.shape)  # (100, 2)
```

With the hypervolume indicator:

```python
result = optimize(
    "zdt2",
    algorithm="ibea",
    max_evaluations=20000,
    seed=42,
    algorithm_kwargs={"indicator": "hv", "kappa": 0.05},
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Population size. |
| `indicator` | `"eps"` | Quality indicator: `"eps"` (additive epsilon, fast) or `"hv"` (hypervolume, slower but more accurate). |
| `kappa` | 0.05 | Fitness scaling factor. Smaller values apply stronger selection pressure. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |

---

## When to use

**Use IBEA when:**

- You want to directly optimize a specific quality indicator rather than using it only for evaluation.
- Fast convergence toward a dense front region is more important than broad initial exploration.
- You are comparing indicator-based and dominance-based approaches experimentally.

**Consider alternatives when:**

- Population sizes are large — fitness recomputation scales quadratically with `pop_size`.
- You have 4+ objectives — try [NSGA-III](nsgaiii.md) or [RVEA](rvea.md).
- Objective scale varies significantly across runs — the `kappa` parameter requires tuning.

---

## Reference

Zitzler, E., & Künzli, S. (2004). Indicator-based selection in multiobjective search. *Parallel Problem Solving from Nature – PPSN VIII*, Lecture Notes in Computer Science, vol. 3242, pp. 832–842. <https://doi.org/10.1007/978-3-540-30217-9_84>

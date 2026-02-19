# SMS-EMOA

S-Metric Selection Evolutionary Multi-Objective Algorithm. A steady-state algorithm that selects for removal the individual contributing least to the population's hypervolume. This directly maximizes the hypervolume indicator, producing fronts with high density in the most valuable regions of objective space.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Hypervolume indicator (steady-state, (μ+1) selection) |
| Objectives | 2–3 (exact HV computation is exponential in objectives) |
| Ask/Tell | Yes |
| Best use case | Problems where front density and hypervolume quality are critical |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="smsemoa", max_evaluations=10000, seed=42)

print(result.F.shape)  # (100, 2)
```

With explicit parameters:

```python
result = optimize(
    "zdt2",
    algorithm="smsemoa",
    max_evaluations=20000,
    seed=42,
    algorithm_kwargs={"pop_size": 100, "crossover_eta": 20},
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
| `ref_point` | auto | Hypervolume reference point. Auto-set as `nadir + 1.1 * range` if not provided. |

---

## When to use

**Use SMS-EMOA when:**

- The final hypervolume value is the primary quality criterion.
- You have 2–3 objectives (exact HV computation is feasible).
- You want a front that is denser in regions with large HV contributions.

**Consider alternatives when:**

- You have 4+ objectives — HV computation becomes exponentially expensive.
- Speed is critical — the steady-state update is slower per generation than NSGA-II's generational update.
- You need a very specific reference-point structure — try [NSGA-III](nsgaiii.md).

---

## Reference

Beume, N., Naujoks, B., & Emmerich, M. (2007). SMS-EMOA: Multiobjective selection based on dominated hypervolume. *European Journal of Operational Research*, 181(3), 1653–1669. <https://doi.org/10.1016/j.ejor.2006.08.008>

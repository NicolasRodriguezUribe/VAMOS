# Algorithms

VAMOS includes nine multi-objective evolutionary algorithms. All share the
same `optimize()` interface. Use public typed configuration objects from
`vamos.algorithms` for algorithm-specific settings.

---

## Selection guide

| Algorithm | Objectives | Paradigm | Highlight |
|-----------|:----------:|----------|-----------|
| [NSGA-II](nsgaii.md) | 2–3 | Dominance + crowding | Fast, widely cited baseline |
| [NSGA-III](nsgaiii.md) | 3–10+ | Reference point | Best for 4+ objectives |
| [MOEA/D](moead.md) | 2–5 | Decomposition | Memory-efficient, scalable |
| [SMS-EMOA](smsemoa.md) | 2–3 | Hypervolume indicator | High front quality |
| [SPEA2](spea2.md) | 2–3 | Archive-based dominance | Clean, fixed-size archive |
| [IBEA](ibea.md) | 2–3 | Quality indicator | Direct indicator optimization |
| [SMPSO](smpso.md) | 2–3 | Particle swarm | Fast on smooth fronts |
| [AGE-MOEA](agemoea.md) | 2–10+ | Adaptive geometry | No reference points needed |
| [RVEA](rvea.md) | 5–15 | Reference vector | Many-objective, adaptive |

---

## Decision guide

**Use NSGA-II** for 2–3 objectives when you want a fast, heavily-tested baseline with a large comparison literature.

**Use NSGA-III** when you have 4 or more objectives and want structured diversity via reference points.

**Use MOEA/D** when the problem has a known decomposable structure and you want efficient neighborhood-based search.

**Use SMS-EMOA** when the quality of the Pareto front (hypervolume) is the primary concern and you have 2–3 objectives.

**Use AGE-MOEA** when you don't know the geometry of the Pareto front in advance. No reference point configuration needed.

**Use RVEA** for 5–15 objectives with adaptive reference vectors that track the current front geometry.

---

## Unified interface

All algorithms are invoked the same way:

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii",   max_evaluations=10000, seed=42)
result = optimize("dtlz2", algorithm="nsgaiii", max_evaluations=50000, seed=42)
result = optimize("zdt3",  algorithm="moead",   max_evaluations=20000, seed=42)
```

For a fully specified configuration:

```python
from vamos.algorithms import NSGAIIConfig

config = NSGAIIConfig.default(pop_size=200, n_var=30)
result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=10000,
    seed=42,
    algorithm_config=config,
)
```

# NSGA-II

Non-dominated Sorting Genetic Algorithm II. The most widely cited multi-objective EA in research and practice. Fast non-dominated sorting combined with a crowding distance operator produces well-spread Pareto front approximations without requiring any additional parameters beyond population size.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Dominance ranking + crowding distance |
| Objectives | 2–3 (degrades on 4+) |
| Ask/Tell | Yes |
| Best use case | General-purpose 2–3 objective optimization |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)

print(result.F.shape)  # (100, 2)
print(result.X.shape)  # (100, 30)
```

Custom problem:

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pop_size` | 100 | Population size. Larger values improve diversity but increase cost per generation. |
| `crossover_prob` | 1.0 | SBX crossover probability. |
| `crossover_eta` | 20 | SBX distribution index. Higher = offspring closer to parents. |
| `mutation_eta` | 20 | Polynomial mutation distribution index. |
| `offspring_size` | `pop_size` | Number of offspring per generation. |

Pass via `algorithm_kwargs`:

```python
result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42,
    algorithm_kwargs={"pop_size": 200, "crossover_eta": 30},
)
```

---

## When to use

**Use NSGA-II when:**

- You have 2–3 objectives and want a reliable, well-tested baseline.
- You need a large comparison literature (thousands of papers use NSGA-II results).
- The problem has a continuous, binary, integer, or permutation encoding.
- You want a fast algorithm with minimal configuration.

**Consider alternatives when:**

- You have 4 or more objectives — try [NSGA-III](nsgaiii.md), [RVEA](rvea.md), or [AGE-MOEA](agemoea.md).
- You need tight hypervolume guarantees — try [SMS-EMOA](smsemoa.md).
- The problem is continuous and you want swarm-based search — try [SMPSO](smpso.md).

---

## Reference

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197. <https://doi.org/10.1109/4235.996017>

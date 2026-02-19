# SMPSO

Speed-Constrained Multi-objective Particle Swarm Optimization. A particle swarm optimizer adapted for multi-objective problems with velocity clamping to prevent divergence. Leaders for the social component are drawn from a crowded-distance external archive. Converges fast on smooth, continuous Pareto fronts.

---

## Key characteristics

| Property | Value |
|----------|-------|
| Paradigm | Swarm intelligence (PSO with velocity clamping) |
| Objectives | 2–3 |
| Ask/Tell | Yes |
| Best use case | Continuous problems with smooth, well-connected Pareto fronts |

---

## Minimal example

```python
from vamos import optimize

result = optimize("zdt1", algorithm="smpso", max_evaluations=10000, seed=42)

print(result.F.shape)  # (100, 2)
```

With explicit swarm parameters:

```python
result = optimize(
    "zdt2",
    algorithm="smpso",
    max_evaluations=20000,
    seed=42,
    algorithm_kwargs={"swarm_size": 100, "leader_size": 100, "w": 0.4},
)
```

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swarm_size` | 100 | Number of particles. |
| `leader_size` | 100 | External archive (leaders) capacity. |
| `w` | 0.4 | Inertia weight. Controls momentum of velocity update. |
| `c1` | 1.5 | Cognitive acceleration factor (attraction toward personal best). |
| `c2` | 1.5 | Social acceleration factor (attraction toward global best leaders). |
| `mutation_eta` | 20 | Polynomial mutation distribution index for perturbation. |

---

## When to use

**Use SMPSO when:**

- Variables are continuous and the landscape is smooth.
- The Pareto front is well-connected (not highly disconnected).
- You want fast initial convergence rather than exhaustive exploration.

**Consider alternatives when:**

- Variables are discrete, binary, or permutation-based — PSO velocity updates assume a continuous space.
- The problem is highly multimodal — SMPSO may stagnate in local fronts.
- You have 4+ objectives — try [NSGA-III](nsgaiii.md) or [AGE-MOEA](agemoea.md).

---

## Reference

Nebro, A. J., Durillo, J. J., Garcia-Nieto, J., Coello Coello, C. A., Luna, F., & Alba, E. (2009). SMPSO: A new PSO-based metaheuristic for multi-objective optimization. *2009 IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making (MCDM)*, pp. 66–73. <https://doi.org/10.1109/MCDM.2009.4938830>

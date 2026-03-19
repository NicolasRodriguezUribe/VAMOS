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
| `archive_mode` | `off` | `off` keeps baseline NSGA-II, `passive` adds an exact unbounded archive without changing mating/survival, `hybrid_survival` changes only split-front truncation on the supported path. |
| `archive_subset_size` | `None` | Size of the exported archive approximation subset; defaults to `pop_size`. |
| `archive_hybrid_alpha` | `0.5` | Hybrid split-front weight on normalized local crowding; archive novelty gets `1 - alpha`. |
| `archive_hybrid_k` | `3` | Archive novelty uses the distance to the `k`-th nearest archived point in normalized objective space. |
| `archive_hybrid_normalization` | `minmax_archive_split` | Current hybrid normalization mode; uses min-max scaling over archive plus split-front objectives. |

Pass via `algorithm_kwargs`:

```python
result = optimize(
    "zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42,
    algorithm_kwargs={"pop_size": 200, "crossover_eta": 30},
)
```

Archive-family modes are opt-in and keep the primary returned result compatible with the standard NSGA-II population semantics. Archive artifacts are exported in `result.data["archive"]`, and compact execution diagnostics are exported in `result.data["archive_diagnostics"]`.

```python
from vamos.algorithms import NSGAIIConfig
from vamos import optimize

cfg = (
    NSGAIIConfig.builder()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("polynomial", prob="1/n", eta=20.0)
    .archive_mode("passive")
    .build()
)

result = optimize("zdt1", algorithm="nsgaii", algorithm_config=cfg, max_evaluations=10000, seed=42)
archive = result.data["archive"]
diagnostics = result.data["archive_diagnostics"]
archive_subset = archive["subset"]
```

## Archive Modes

`archive_mode="off"`:

- Baseline NSGA-II behavior.
- Standard mating and standard kernel survival.
- No archive-family behavior is applied.

`archive_mode="passive"`:

- Maintains an exact unbounded external archive across the run.
- Does not affect mating.
- Does not affect survival.
- Exports the full archive, `archive["size"]`, and a crowding-selected subset at `archive["subset"]`.

`archive_mode="hybrid_survival"`:

- Keeps Pareto rank and complete-front acceptance unchanged.
- Changes only split-front truncation.
- Uses a hybrid score on the split front: normalized crowding plus normalized archive novelty.
- Does not affect mating.
- Uses the historical archive as the novelty reference.

## Current Limitations

- `hybrid_survival` is currently active only for the standard generational unconstrained NSGA-II path.
- Constrained and steady-state/incremental runs fall back to the standard survival path.
- Archive-assisted mating is not implemented.
- The current archive subset export uses crowding-based selection.

When `hybrid_survival` falls back, or when the split front uses local-only scoring because the archive is missing or too small for `k`, the final result records that in `result.data["archive_diagnostics"]`.

## Exported Archive Artifacts

- `result.data["archive"]["F"]`: full archive objective matrix.
- `result.data["archive"]["X"]`: full archive decision matrix when available.
- `result.data["archive"]["G"]`: full archive constraint matrix when available.
- `result.data["archive"]["size"]`: full archive size.
- `result.data["archive"]["subset"]`: crowding-selected approximation subset of size `archive_subset_size` or `pop_size` by default.
- `result.data["archive_diagnostics"]`: mode, survival path, fallback reason, and hybrid split-front traceability.

For a compact runnable demo, see `examples/advanced/nsgaii_archive_modes.py`.

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

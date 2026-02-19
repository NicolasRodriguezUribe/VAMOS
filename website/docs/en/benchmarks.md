# Benchmarks

Runtime and solution quality comparisons against pymoo, DEAP, jMetalPy, and Platypus. All comparisons use NSGA-II on a single CPU core.

---

## Setup

| Setting | Value |
|---------|-------|
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| CPU | Intel Core i7-12700K (single core, P-core, no hyperthreading) |
| OS | Ubuntu 22.04 LTS |
| Algorithm | NSGA-II, population 100 |
| Metric | Median of 10 independent runs |

Timing covers initialization, the full optimization loop, and result extraction. Python import time is excluded.

---

## Runtime comparison

### ZDT1 — 2 objectives, 30 variables

100 population · 200 generations · **20 000 evaluations**

| Framework | Median time | Relative to VAMOS |
|-----------|:-----------:|:-----------------:|
| **VAMOS** | **0.41 s** | 1.0× |
| pymoo 0.6.1 | 0.50 s | 1.22× |
| DEAP 1.4.1 | 2.12 s | 5.2× |
| jMetalPy 1.7.0 | 3.84 s | 9.4× |
| Platypus 1.2.1 | 1.91 s | 4.7× |

### DTLZ2 — 3 objectives, 12 variables

100 population · 500 generations · **50 000 evaluations**

| Framework | Median time | Relative to VAMOS |
|-----------|:-----------:|:-----------------:|
| **VAMOS** | **1.23 s** | 1.0× |
| pymoo 0.6.1 | 1.47 s | 1.20× |
| DEAP 1.4.1 | 7.91 s | 6.4× |
| jMetalPy 1.7.0 | 14.3 s | 11.6× |
| Platypus 1.2.1 | 6.22 s | 5.1× |

### ZDT4 — 2 objectives, 10 variables (multimodal)

100 population · 500 generations · **50 000 evaluations**

| Framework | Median time | Relative to VAMOS |
|-----------|:-----------:|:-----------------:|
| **VAMOS** | **1.02 s** | 1.0× |
| pymoo 0.6.1 | 1.22 s | 1.20× |
| DEAP 1.4.1 | 5.84 s | 5.7× |
| jMetalPy 1.7.0 | 9.61 s | 9.4× |
| Platypus 1.2.1 | 4.88 s | 4.8× |

---

## Where the speedup comes from

DEAP, jMetalPy, and Platypus iterate over individuals in a Python `for` loop. Each evaluation invokes a Python function call — significant overhead when the objective itself is cheap (ZDT1 runs in nanoseconds per individual).

VAMOS builds a vectorized evaluation function at problem-creation time. The entire population is evaluated in a single NumPy call, eliminating Python-level iteration.

pymoo uses the same vectorization strategy, which is why VAMOS is only ~1.2× faster than pymoo rather than 5–12×.

Optional Numba acceleration (`engine="numba"`) provides an additional speedup for problems with large populations or expensive operator computations.

---

## Solution quality

Speed is meaningless if solution quality suffers. The table below reports **Inverted Generational Distance (IGD)** — lower is better. Results are the median of 10 independent seeds.

NSGA-II · ZDT1 · 100 pop · 200 generations

| Framework | Median IGD | Std IGD |
|-----------|:----------:|:-------:|
| **VAMOS** | **4.02e-3** | 1.1e-4 |
| pymoo | 4.01e-3 | 1.2e-4 |
| DEAP | 4.15e-3 | 3.4e-4 |
| jMetalPy | 4.09e-3 | 2.0e-4 |
| Platypus | 4.18e-3 | 3.8e-4 |

All frameworks reach equivalent IGD when using the same NSGA-II logic. Differences are within statistical noise. VAMOS does not trade solution quality for speed.

---

## ZDT benchmark suite — NSGA-II quality

Median IGD across 10 seeds · 100 pop · 10 000 evaluations per problem

| Problem | Median IGD | Description |
|---------|:----------:|-------------|
| ZDT1 | 4.02e-3 | Convex front, 30 vars |
| ZDT2 | 3.98e-3 | Concave front, 30 vars |
| ZDT3 | 5.11e-3 | Disconnected front, 30 vars |
| ZDT4 | 2.31e-2 | Multimodal landscape, 10 vars |
| ZDT6 | 1.87e-2 | Nonuniform density, 10 vars |

Reproduce with:

```python
from vamos import optimize
import numpy as np

for problem in ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]:
    results = [
        optimize(problem, algorithm="nsgaii", max_evaluations=10000, seed=s)
        for s in range(10)
    ]
    igd_vals = [r.meta["igd"] for r in results]
    print(f"{problem}: IGD = {np.median(igd_vals):.2e} ± {np.std(igd_vals):.1e}")
```

---

## Numba acceleration

Install the `numba` extra and pass `engine="numba"` for JIT-compiled operators:

```python
result = optimize(
    "zdt1", algorithm="nsgaii",
    max_evaluations=100000, engine="numba", seed=42,
)
```

The first call includes compilation overhead (~2–5 s). Subsequent calls on the same problem structure are faster. Numba benefits are largest for large populations (≥500) and high-dimensional problems.

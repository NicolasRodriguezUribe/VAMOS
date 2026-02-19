---
title: VAMOS — Multi-Objective Evolutionary Optimization
description: Fast. Clean. Multi-objective. Python framework for multi-objective evolutionary optimization.
hide:
  - navigation
  - toc
---

<div class="vamos-hero" markdown>

# Fast. Clean.<br>Multi-objective.

Multi-objective evolutionary optimization in Python.<br>
Nine algorithms. Two-line API. Vectorized core.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[Browse Algorithms](algorithms/index.md){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/NicolasRodriguezUribe/VAMOS){ .md-button }

</div>

<div class="vamos-install" markdown>

## Install

```bash
pip install vamos-optimization
```

Python 3.9+ &nbsp;·&nbsp; Core: `numpy` · `scipy` · `joblib` — nothing else required &nbsp;·&nbsp; Optional extras: `numba` · `pandas` · `matplotlib` · `streamlit` · `optuna`

</div>

## Quick Start

=== "One-liner"

    Run any built-in benchmark with a single call.

    ```python
    from vamos import optimize

    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)
    ```

=== "Custom problem"

    Wrap any Python function. Pass variable count, objective count, and bounds.

    ```python
    from vamos import make_problem, optimize

    problem = make_problem(
        lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
        n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
    )
    result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
    ```

=== "Results"

    Pareto front and decision variables as NumPy arrays.

    ```python
    result.F   # objective values, shape (N, n_obj)
    result.X   # decision variables, shape (N, n_var)
    ```

---

## Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Vectorized Core**

    ---

    Entire populations evaluated in a single NumPy call. No Python loops over individuals. Scales to large populations with no per-individual overhead.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-gauge:{ .lg .middle } **9 Algorithms**

    ---

    NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA — consistent API across all nine.

    [:octicons-arrow-right-24: Algorithms](algorithms/index.md)

-   :material-code-braces:{ .lg .middle } **Two-line API**

    ---

    `from vamos import optimize`, then call it. No mandatory subclassing. Lambda functions work as problem definitions.

    [:octicons-arrow-right-24: Getting Started](getting-started.md)

-   :material-tune:{ .lg .middle } **Built-in Tuning**

    ---

    Optuna-backed hyperparameter search with pre-built config spaces for every algorithm. Start tuning in three lines.

    [:octicons-arrow-right-24: Tuning Tutorial](tutorials/tuning.md)

-   :material-monitor-dashboard:{ .lg .middle } **VAMOS Studio**

    ---

    Interactive Streamlit dashboard for problem setup, Pareto front visualization, and solution comparison. Install with `pip install vamos-optimization[studio]`.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-chart-line:{ .lg .middle } **Reproducible Benchmarks**

    ---

    Fixed seeds, structured result objects, built-in hypervolume and IGD tracking. Every run reproducible by default.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

</div>

---

## Performance

VAMOS is **4–12× faster** than DEAP, jMetalPy, and Platypus, and **~1.1–1.2× faster** than pymoo. The speedup comes from NumPy-vectorized population operators — no JIT compiler needed for the baseline numbers.

| Framework | ZDT1 · 20 000 evals | DTLZ2 (3-obj) · 50 000 evals | Relative to VAMOS |
|-----------|:---:|:---:|:---:|
| **VAMOS** | **0.41 s** | **1.23 s** | **1.0×** |
| pymoo 0.6 | 0.50 s | 1.47 s | ~1.2× |
| DEAP 1.4 | 2.12 s | 7.91 s | 5–6× |
| jMetalPy 1.7 | 3.84 s | 14.3 s | 9–12× |
| Platypus 1.2 | 1.91 s | 6.22 s | 4–5× |

*NSGA-II · 100 population · single core · Intel Core i7-12700K · Python 3.11 · smaller is better.*

---

## Why VAMOS?

**No boilerplate.** Most frameworks require subclassing a `Problem` class and overriding specific methods. VAMOS accepts plain Python functions — including lambdas.

**Vectorization by default.** Write `f(x) → [f1, f2]`. VAMOS evaluates the full population in one NumPy call internally. You never write broadcasting code for your objectives.

**Minimal dependencies.** The core package needs only `numpy`, `scipy`, and `joblib`. No autodiff engine, no distributed infrastructure pulled in as a mandatory dependency.

**Reproducibility first.** Pass `seed=42` and get the same result on any machine. Result objects carry the algorithm name, full configuration, and evaluation count.

**Tuning included.** Every algorithm ships with a validated Optuna config space. Racing and model-based tuners are built in — not bolted on.

# VAMOS

<div class="vamos-hero">
  <h1 class="vamos-hero__tagline">Fast. Clean. <span>Multi-objective.</span></h1>
  <p class="vamos-hero__subtitle">The multi-objective optimization framework that gets out of your way.</p>
  <div class="vamos-hero__install">
    <span>$</span>
    <code>pip install vamos-optimization</code>
  </div>
  <div class="vamos-hero__actions">
    <a href="getting-started/" class="vamos-btn vamos-btn--primary">Get Started</a>
    <a href="https://github.com/vamos-optimization/vamos" class="vamos-btn vamos-btn--outline">GitHub</a>
  </div>
</div>

## Solve in two lines

=== "One-liner"

    ```python
    from vamos import optimize

    result = optimize("zdt1", algorithm="nsgaii", max_evaluations=10000, seed=42)
    print(f"Found {len(result.F)} Pareto-optimal solutions")
    ```

=== "Custom problem"

    ```python
    from vamos import make_problem, optimize

    problem = make_problem(
        lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
        n_var=2, n_obj=2, bounds=[(0, 1), (0, 1)],
    )
    result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
    ```

=== "Results"

    ```python
    result.F   # objective values — shape (N, n_obj)
    result.X   # decision variables — shape (N, n_var)

    print(result.F[:, 0].min())   # best f1 value
    print(result.F[:, 1].max())   # worst f2 value
    ```

---

## Why VAMOS?

Most multi-objective optimization frameworks make you choose between performance and usability. Low-level frameworks (DEAP, Platypus) give you control but require 20+ lines of boilerplate per experiment. Higher-level ones (pymoo) are cleaner but still ask you to subclass, instantiate, and wire components together manually.

VAMOS takes a different approach: population data lives in dense arrays (`X ∈ ℝ^(N×n)`, `F ∈ ℝ^(N×m)`), hot loops dispatch to vectorized NumPy/Numba kernels, and the entire API surface fits in a single `optimize()` call. The result is a framework that runs 4–12× faster than object-centric alternatives and requires a fraction of the setup code — without sacrificing flexibility when you need it.

---

## Features

<div class="vamos-cards">
  <div class="vamos-card">
    <div class="vamos-card__icon">⚡</div>
    <div class="vamos-card__title">Vectorized Core</div>
    <p class="vamos-card__body">Dense array populations dispatched to NumPy, Numba, or MooCore kernels. 4–12× faster than DEAP, jMetalPy, and Platypus. ~1.1–1.2× faster than pymoo.</p>
  </div>
  <div class="vamos-card">
    <div class="vamos-card__icon">🧬</div>
    <div class="vamos-card__title">9 Algorithms</div>
    <p class="vamos-card__body">NSGA-II, NSGA-III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA — all fully implemented with Ask/Tell support in 7 of 9.</p>
  </div>
  <div class="vamos-card">
    <div class="vamos-card__icon">✍️</div>
    <div class="vamos-card__title">Two-line API</div>
    <p class="vamos-card__body"><code>optimize("zdt1", algorithm="nsgaii")</code> — vs. ~10 lines in pymoo, ~20 in DEAP. <code>make_problem(fn)</code> auto-vectorizes any scalar function.</p>
  </div>
  <div class="vamos-card">
    <div class="vamos-card__icon">🧪</div>
    <div class="vamos-card__title">Built-in Tuning</div>
    <p class="vamos-card__body">Multi-fidelity racing with warm-start checkpoints. Pluggable backends: racing, random, Optuna, SMAC3, BOHB. Run via <code>vamos tune</code>.</p>
  </div>
  <div class="vamos-card">
    <div class="vamos-card__icon">🖥️</div>
    <div class="vamos-card__title">VAMOS Studio</div>
    <p class="vamos-card__body">Browser-based interactive dashboard. Visual problem builder, live Pareto front preview, and MCDM tools. No Python knowledge required.</p>
  </div>
  <div class="vamos-card">
    <div class="vamos-card__icon">📊</div>
    <div class="vamos-card__title">Reproducible Benchmarks</div>
    <p class="vamos-card__body">Semantic-alignment protocol for fair cross-framework comparisons. Paired Wilcoxon tests with Holm correction included out of the box.</p>
  </div>
</div>

---

## Performance

Runtime comparison on ZDT1, 10,000 evaluations, NSGA-II equivalent, averaged over 30 runs.

<table class="vamos-perf-table">
  <thead>
    <tr>
      <th>Framework</th>
      <th>Time (s)</th>
      <th>vs VAMOS</th>
      <th>API complexity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>VAMOS</strong></td>
      <td class="highlight">1.0×</td>
      <td><span class="vamos-badge vamos-badge--fastest">Baseline</span></td>
      <td>2 lines</td>
    </tr>
    <tr>
      <td>pymoo</td>
      <td>1.1–1.2×</td>
      <td><span class="vamos-badge vamos-badge--good">Close</span></td>
      <td>~10 lines</td>
    </tr>
    <tr>
      <td>DEAP</td>
      <td>4–12×</td>
      <td><span class="vamos-badge vamos-badge--slow">Slower</span></td>
      <td>~20 lines</td>
    </tr>
    <tr>
      <td>jMetalPy</td>
      <td>4–12×</td>
      <td><span class="vamos-badge vamos-badge--slow">Slower</span></td>
      <td>~15 lines</td>
    </tr>
    <tr>
      <td>Platypus</td>
      <td>4–12×</td>
      <td><span class="vamos-badge vamos-badge--slow">Slower</span></td>
      <td>~12 lines</td>
    </tr>
  </tbody>
</table>

---

## Quick links

- [**Getting Started**](getting-started.md) — install and run your first optimization in 5 minutes
- [**Algorithms**](algorithms/index.md) — choose the right algorithm for your problem
- [**API Reference**](api/index.md) — full `optimize()`, `make_problem()`, and `OptimizationResult` docs
- [**Tutorials**](tutorials/quickstart.md) — hands-on notebooks from beginner to advanced

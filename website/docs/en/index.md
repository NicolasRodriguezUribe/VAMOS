# VAMOS 1.0.0

<div class="vamos-hero">
  <h1 class="vamos-hero__tagline">Reproducible <span>multi-objective optimization.</span></h1>
  <p class="vamos-hero__subtitle">A typed Python API, vectorized kernels, and durable run and study artifacts.</p>
  <div class="vamos-hero__install">
    <span>$</span>
    <code>pip install vamos-optimization</code>
  </div>
  <div class="vamos-hero__actions">
    <a href="getting-started/" class="vamos-btn vamos-btn--primary">Get Started</a>
    <a href="https://github.com/vamos-optimization/VAMOS" class="vamos-btn vamos-btn--outline">GitHub</a>
  </div>
</div>

VAMOS 1.0.0 is the first official public release and compatibility baseline.
Earlier version strings and tags were internal pre-public development markers,
not prior public releases.

## Run an optimization

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=400,
    pop_size=40,
    engine="numpy",
    seed=42,
)

print(result.F.shape)
print(result.data["evaluations"])
```

## What the release provides

- Nine built-in multi-objective algorithms behind one stable `optimize()`
  entry point.
- NumPy as the deterministic reference backend, plus optional Numba kernel and
  MooCore indicator acceleration.
- Scalar and explicitly vectorized custom-problem adapters.
- Canonical, bounded run artifacts with data-only loading and verification.
- Exact same-environment replay for reconstructable registered built-ins.
- A durable single-owner study lifecycle with planning, execution, inspection,
  summary, resume, and retry.
- An experimental local Studio with explicit trusted-code consent.

Performance and scientific-quality claims depend on the problem, configuration,
environment, and comparison protocol. VAMOS therefore ships reproducible
benchmark tooling instead of asserting a universal speedup.

## Quick links

- [Getting Started](getting-started.md)
- [Algorithms](algorithms/index.md)
- [API Reference](api/index.md)
- [Quickstart Tutorial](tutorials/quickstart.md)
- [Benchmark methodology](benchmarks.md)

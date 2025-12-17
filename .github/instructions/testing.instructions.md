---
applyTo: "tests/**/*.py"
description: Testing guidelines for VAMOS
---

# Testing Standards for VAMOS

## Test Organization
- Mirror `src/vamos/` structure under `tests/` (foundation, engine, experiment, ux, integration)
- Use `pytest` markers: `@pytest.mark.slow`, `@pytest.mark.smoke`, `@pytest.mark.backends`, `@pytest.mark.examples`, `@pytest.mark.cli`

## Smoke Tests (Algorithms)
- Keep budgets tiny: `pop_size=10`, `max_evaluations=100`
- Assert fronts are non-empty and finite: `assert F.shape[0] > 0 and np.isfinite(F).all()`

## Deterministic vs Stochastic
- Unit tests: deterministic components (operators, kernels, registries)
- Stochastic algorithms: use statistical/property checks, not exact value matching
- Always seed with `np.random.default_rng(seed)` for reproducibility

## Example Smoke Test Pattern
```python
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.core.runner import run_single, ExperimentConfig

def test_nsgaii_zdt1_smoke():
    selection = make_problem_selection("zdt1", n_var=10)
    cfg = ExperimentConfig(population_size=10, max_evaluations=100, seed=42)
    result = run_single("numpy", "nsgaii", selection, cfg)
    assert result["F"].shape[0] > 0
    assert np.isfinite(result["F"]).all()
```

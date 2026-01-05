# Racing Tuner Package

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates.

## Overview

This package implements **irace-inspired algorithm configuration** with modern enhancements:
- Multi-fidelity evaluation (Hyperband-style successive halving)
- Warm-starting between fidelity levels
- Model-based sampling
- Parallel evaluation with joblib

## Structure

### Core Files
- `core.py`: `RacingTuner` - main racing loop with multi-fidelity support
- `scenario.py`: `Scenario` - racing strategy configuration
- `tuning_task.py`: `TuningTask`, `EvalContext`, `Instance` - task definition
- `state.py`: `ConfigState`, `EliteEntry` - internal state tracking

### Multi-Fidelity Support
- `warm_start.py`: `WarmStartEvaluator` - helper for checkpoint-based continuation
- Key `Scenario` parameters:
  - `use_multi_fidelity: bool` - enable Hyperband-style racing
  - `fidelity_levels: tuple[int, ...]` - budget sequence (e.g., `(1000, 3000, 10000)`)
  - `fidelity_warm_start: bool` - enable checkpoint passing between levels
  - `fidelity_promotion_ratio: float` - fraction promoted to next level

### Parameter Space
- `param_space.py`: `ParamSpace`, `Real`, `Int`, `Categorical`, `Boolean`, `Condition`
- `sampler.py`: `UniformSampler`, `ModelBasedSampler`
- `config_space.py`: `AlgorithmConfigSpace` - algorithm-specific space builders

### Utilities
- `bridge.py`: `build_nsgaii_config_space()`, `config_from_assignment()` - algorithm integration
- `elimination.py`: Statistical elimination logic (Friedman/Wilcoxon tests)
- `io.py`: Checkpoint save/load, history export (JSON, CSV)

## Usage Pattern

```python
from vamos.engine.tuning.racing import (
    RacingTuner, Scenario, TuningTask, Instance,
    WarmStartEvaluator, build_nsgaii_config_space
)

# 1. Define run function
def run_algo(config, ctx, checkpoint=None):
    # ... run algorithm ...
    return result, new_checkpoint

# 2. Create evaluator
evaluator = WarmStartEvaluator(run_fn=run_algo, score_fn=...)

# 3. Configure
scenario = Scenario(
    max_experiments=300,
    use_multi_fidelity=True,
    fidelity_levels=(1000, 3000, 10000),
)

# 4. Run
tuner = RacingTuner(task, scenario)
best, history = tuner.run(evaluator)
```

## Conventions

- `EvalContext.checkpoint` carries algorithm state between fidelity levels
- Eval functions return `(score, checkpoint)` tuple when warm-starting
- Scores are aggregated via `TuningTask.aggregator` (default: `np.mean`)
- Higher scores are better when `maximize=True`

## Adding New Features

1. Add parameters to `Scenario` with documentation
2. Add validation in `Scenario.__post_init__`
3. Implement logic in `RacingTuner._run_*` methods
4. Export from `__init__.py`
5. Add tests under `tests/engine/tuning/`

# Algorithm Module

This directory contains the evolutionary algorithm cores for VAMOS.

## Architecture

All algorithms follow the **ask-tell** pattern with array-based populations:
- `ask()` → returns offspring decision variables `X_off` (N × D array)
- `tell(eval_result)` → updates internal state with evaluated offspring

Populations are **never** per-individual objects. Use arrays:
- `X` — decision variables (pop_size × n_var)
- `F` — objective values (pop_size × n_obj)
- `G` — constraint violations (pop_size × n_constraints), optional

## Adding a New Algorithm

1. Create `new_algo.py` implementing the class with `__init__(config, kernel)` and `run(problem, termination, seed, ...)`
2. Add config dataclass + builder in [config.py](config.py) following the fluent pattern
3. Register in [registry.py](registry.py): `ALGORITHMS["new_algo"] = lambda cfg, kernel: NewAlgo(cfg, kernel)`
4. Add smoke test in `tests/test_algorithms_smoke.py`

## Key Files

| File | Purpose |
|------|---------|
| `nsgaii.py` | NSGA-II core (reference implementation) |
| `config.py` | Config dataclasses + fluent builders |
| `registry.py` | Algorithm name → builder mapping |
| `variation.py` | VariationPipeline for crossover/mutation |
| `archive.py` | HypervolumeArchive, CrowdingDistanceArchive |
| `selection.py` | Tournament, random selection |

## Config Builder Pattern

```python
from vamos.algorithm.config import NSGAIIConfig

cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", pressure=2)
    .survival("nsga2")
    .engine("numpy")
    .fixed())  # Returns immutable NSGAIIConfigData
```

## Kernel Dependency

Algorithms delegate heavy operations to `self.kernel`:
- `kernel.fast_non_dominated_sort(F)` → ranks, crowding
- `kernel.hypervolume(F, ref)` → HV indicator
- `kernel.nsga2_ranking(F)` → combined rank + crowding

Never implement sorting/HV in algorithm code directly.

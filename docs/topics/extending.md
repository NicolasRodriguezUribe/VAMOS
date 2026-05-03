# Extending VAMOS

This is the canonical extension guide. The developer pages for algorithms and backends are intentionally thin and defer here.

## Extension paths

- Add a custom algorithm when you need a new search loop or survival/update strategy.
- Add a custom operator when the algorithm can stay the same but variation or repair changes.
- Add a custom kernel backend only for performance-critical numeric primitives that already have a stable NumPy reference path.
- Add a custom problem when the framework logic is fine and only the optimization target changes.

## Algorithms

Implement the algorithm class under `src/vamos/engine/algorithm/`, then register a builder in the global registry. This minimal path is the canonical, smoke-tested extension workflow:

```python
from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.registry import get_algorithms_registry
from vamos.foundation.kernel.backend import KernelBackend


class MyAlgorithm:
    def __init__(self, config: dict[str, Any], kernel: KernelBackend | None = None) -> None:
        self.config = config
        self.kernel = kernel

    def run(self, problem, termination=("max_evaluations", 100), seed=0, eval_strategy=None, live_viz=None):
        return {
            "X": np.empty((0, problem.n_var)),
            "F": np.empty((0, problem.n_obj)),
            "evaluations": 0,
        }


def build_my_algorithm(cfg: dict[str, Any], kernel: KernelBackend | None = None) -> MyAlgorithm:
    return MyAlgorithm(dict(cfg), kernel=kernel)


get_algorithms_registry().register("my_algorithm", build_my_algorithm)
```

Installed packages can expose algorithms without import-time side effects by
declaring a `vamos.algorithms` entry point whose name is the algorithm key and
whose value is the builder callable:

```toml
[project.entry-points."vamos.algorithms"]
my_algorithm = "my_package.vamos_plugin:build_my_algorithm"
```

VAMOS discovers these entry points lazily through `available_algorithms()`,
`resolve_algorithm(...)`, or explicitly with `discover_algorithm_plugins()`.

Rules:

- Keep `optimize(...)` as the public entrypoint; do not add a second facade.
- Add a typed config builder under `src/vamos/engine/algorithm/config/` only when the algorithm needs a stable public configuration object.
- Add a smoke test that resolves the algorithm through the registry and runs a tiny budget.
- If you publish an example in docs, make it registry-based and executable from the docs smoke manifest.

## Operators

- Add operator implementations under `src/vamos/engine/operators/impl/`.
- Register classes or factories through `src/vamos/engine/operators/impl/registry.py`.
- Keep implementations vectorized and RNG-driven. Avoid algorithm-specific conditionals inside generic operators; put those in `src/vamos/engine/operators/policies/`.

## Kernel backends

Kernel backends live under `src/vamos/foundation/kernel/` and are reserved for high-call-count numeric primitives such as ranking, survival, mutation, or selection.

Rules:

- Mirror the NumPy backend interface in `src/vamos/foundation/kernel/backend.py`.
- Keep optional dependencies lazy-loaded through `src/vamos/foundation/kernel/registry.py`.
- Every accelerated kernel must keep a NumPy reference path and parity tests.
- Do not add control-heavy orchestration to kernel backends.

## Problems

- Add problem classes under `src/vamos/foundation/problem/`.
- Register family specs under `src/vamos/foundation/problem/registry/families/`.
- Ensure `evaluate(X, out)` fills `out["F"]` and `out["G"]` in vectorized form when constrained.
- Use `n_constraints` as the only public constraint-count field.

## Config and CLI

- When adding public knobs, update both `src/vamos/experiment/cli/` and the typed config/dataclass layer.
- Keep YAML/JSON spec defaults aligned with CLI defaults.
- Prefer public examples based on `optimize(...)`; show raw config objects only when the extra control matters.

## Tests and docs

- Update the nearest user-facing docs page and keep the example path executable.
- Add pytest coverage near the corresponding module; include determinism and parity checks where relevant.
- Run `ruff check src tests`, `ruff format src tests`, `mypy src/vamos`, and `pytest` before opening a PR.


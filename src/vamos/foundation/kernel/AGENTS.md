# Scope

Applies only to `src/vamos/foundation/kernel/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `KernelBackend` defines the current required primitives: `nsga2_ranking`, `tournament_selection`, `sbx_crossover`, `polynomial_mutation`, and `nsga2_survival`.
- Archive and quality-indicator hooks are optional and must advertise capability before callers use them.
- `NumPyKernel` is the reference semantics. Numba and MooCore implementations are accelerators and require parity with it.
- Optional backend imports remain lazy in `registry.py`; importing core VAMOS must not import optional packages.
- Backends implement numerical hot paths only. Keep orchestration, configuration policy, studies, and CLI logic outside this subtree.

## Extension touchpoints

Follow [Adding a backend](/docs/dev/add_backend.md). Extend the abstract interface only when every supported backend and its parity coverage change together.

## Targeted validation

Run `python -m pytest -q tests/foundation/test_backends_smoke.py tests/foundation/test_numba_backend_parity.py tests/foundation/test_kernel_failures.py tests/engine/test_kernel_selection_dispatch.py` with the relevant optional extra installed.

```agent-docs
path: src/vamos/foundation/kernel/backend.py
path: src/vamos/foundation/kernel/numpy_backend.py
path: src/vamos/foundation/kernel/registry.py
path: tests/foundation/test_backends_smoke.py
path: tests/foundation/test_numba_backend_parity.py
path: tests/foundation/test_kernel_failures.py
path: tests/engine/test_kernel_selection_dispatch.py
path: docs/dev/add_backend.md
symbol: vamos.foundation.kernel.backend:KernelBackend
symbol: vamos.foundation.kernel.registry:resolve_kernel
command: python -m pytest -q tests/foundation/test_backends_smoke.py tests/foundation/test_numba_backend_parity.py tests/foundation/test_kernel_failures.py tests/engine/test_kernel_selection_dispatch.py
```

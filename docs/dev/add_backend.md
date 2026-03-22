# Adding A Kernel Backend

Use [Extending VAMOS](../topics/extending.md) as the canonical workflow.

## Checklist

1. Implement the `KernelBackend` interface in `src/vamos/foundation/kernel/backend.py`.
2. Keep the NumPy backend as the reference behavior; new backends are accelerators, not alternate semantics.
3. Register the backend lazily in `src/vamos/foundation/kernel/registry.py`.
4. Add parity tests against the NumPy backend and a backend-marked smoke test.
5. If the backend uses optional dependencies, keep them out of the default install and return clear import errors.

## Scope guard

Kernel backends are for numeric hot paths only. CLI logic, orchestration, study management, and Studio code should not move into backend implementations.

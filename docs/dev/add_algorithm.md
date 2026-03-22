# Adding A New Algorithm

Use [Extending VAMOS](../topics/extending.md) as the canonical workflow.

## Checklist

1. Implement the algorithm core under `src/vamos/engine/algorithm/`.
2. Register the builder with `get_algorithms_registry().register(...)`.
3. Add a typed config object only if the algorithm needs a stable public configuration surface.
4. Add a registry-based smoke test and any required determinism/parity checks.
5. Update [Extending VAMOS](../topics/extending.md) if the extension workflow itself changed.

## Non-goals

- Do not add a second public optimization facade.
- Do not mutate undocumented globals from examples.
- Do not document a registration path here that differs from [Extending VAMOS](../topics/extending.md).

# Scope

Applies only to `src/vamos/engine/operators/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `impl/` owns reusable vectorized crossover, mutation, repair, and initialization implementations.
- `impl/registry.py` maps stable operator names to classes or factories. `policies/` owns encoding- and algorithm-aware construction decisions.
- `src/vamos/engine/variation/` owns the shared pipeline/protocol that composes these operators for algorithms.
- Operators receive an explicit `numpy.random.Generator`; never read global random state.
- Preserve population shapes, dtypes, bounds, and the semantics of real, integer, binary, permutation, and mixed encodings.
- Keep algorithm-specific branching out of generic implementations. Reuse workspace inputs where an existing protocol supplies them.

## Extension touchpoints

Add the implementation under the matching `impl/` encoding package, export it from that package, register it in `impl/registry.py`, and update the relevant policy only when selection by algorithm/config requires it. Follow [Adding an operator](/docs/dev/add_operator.md).

## Targeted validation

Run `python -m pytest -q tests/engine/operators tests/engine/test_discrete_and_mixed_operators.py tests/engine/test_operator_combo_validation.py` and the smallest algorithm smoke that consumes the operator.

```agent-docs
path: src/vamos/engine/operators/impl/registry.py
path: src/vamos/engine/operators/policies
path: src/vamos/engine/variation
path: tests/engine/operators
path: tests/engine/test_discrete_and_mixed_operators.py
path: tests/engine/test_operator_combo_validation.py
path: docs/dev/add_operator.md
symbol: vamos.engine.operators.impl.registry:get_operator_registry
symbol: vamos.engine.variation:VariationPipeline
command: python -m pytest -q tests/engine/operators tests/engine/test_discrete_and_mixed_operators.py tests/engine/test_operator_combo_validation.py
```

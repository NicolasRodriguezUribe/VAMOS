# Scope

Applies only to `src/vamos/foundation/problem/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- Problem implementations own dimensions, bounds, encoding, and batched evaluation; orchestration does not repair an invalid problem contract.
- `evaluate(X, out)` accepts a two-dimensional population and writes `out["F"]`; constrained problems also write `out["G"]` using `g <= 0` as feasible.
- Use `n_constraints` as the public constraint count. Preserve deterministic evaluation and validate shapes at the problem boundary.
- Keep packaged reference data under `src/vamos/resources/` and access it through resource helpers, not working-directory paths.

## Built-in extension touchpoints

- Put an implementation in the appropriate family module under this subtree.
- Register its `ProblemSpec` in `registry/families/*.py`; `registry/specs.py` only assembles family maps and is not the per-problem edit point.
- Export a class through `vamos.problems` only when it is intentionally public; named registry access is sufficient for most benchmarks.
- Follow [Adding a problem](/docs/dev/add_problem.md) for the exact workflow.

## Targeted validation

Run `python -m pytest -q tests/foundation/test_problem_registry.py tests/foundation/test_problem_zoo.py tests/foundation/test_problem_evaluation_edge_cases.py`, plus a focused test for the affected family.

```agent-docs
path: src/vamos/foundation/problem/registry/families
path: src/vamos/foundation/problem/registry/specs.py
path: src/vamos/resources
path: tests/foundation/test_problem_registry.py
path: tests/foundation/test_problem_zoo.py
path: tests/foundation/test_problem_evaluation_edge_cases.py
path: docs/dev/add_problem.md
command: python -m pytest -q tests/foundation/test_problem_registry.py tests/foundation/test_problem_zoo.py tests/foundation/test_problem_evaluation_edge_cases.py
```

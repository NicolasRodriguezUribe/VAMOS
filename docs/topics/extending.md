# Extending VAMOS

Choose one extension boundary and follow its canonical developer guide:

| Goal | Canonical guide | Owning area |
| --- | --- | --- |
| Define an ad-hoc or built-in problem | [Adding a problem](../dev/add_problem.md) | `foundation/problem` |
| Add crossover, mutation, repair, or initialization | [Adding an operator](../dev/add_operator.md) | `engine/operators` |
| Add a search lifecycle | [Adding an algorithm](../dev/add_algorithm.md) | `engine/algorithm` |
| Accelerate a stable numerical primitive | [Adding a backend](../dev/add_backend.md) | `foundation/kernel` |
| Add a quality indicator | [Adding a metric](../dev/add_metric.md) | `foundation/quality_indicators` |

Changes to [the CLI](../dev/cli.md), [studies](../dev/studies.md), or [run artifacts and exact replay](../dev/run_artifacts_and_replay.md) use their own boundaries. Do not place orchestration in a numerical extension or create a second public facade.

Public examples use `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`. Registry and implementation imports are internal contributor surfaces. Every extension needs a small deterministic or seeded test, a registry/resolution test where applicable, and documentation of the supported public route.

VAMOS is pre-release: replace an internal contract comprehensively rather than adding a second spelling or path for it.

```agent-docs
path: docs/dev/add_problem.md
path: docs/dev/add_operator.md
path: docs/dev/add_algorithm.md
path: docs/dev/add_backend.md
path: docs/dev/add_metric.md
path: docs/dev/cli.md
path: docs/dev/studies.md
path: docs/dev/run_artifacts_and_replay.md
```

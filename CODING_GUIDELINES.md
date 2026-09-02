# Coding guidelines

The repository-wide coding, architecture, Git, security, and validation contract is [AGENTS.md](AGENTS.md). It applies to human and AI-assisted contributions. This page is a discovery index, not a second source of rules.

Use the canonical guide for the change:

- [Adding a problem](docs/dev/add_problem.md)
- [Adding an operator](docs/dev/add_operator.md)
- [Adding an algorithm](docs/dev/add_algorithm.md)
- [Adding a backend](docs/dev/add_backend.md)
- [Adding a metric](docs/dev/add_metric.md)
- [Changing the CLI](docs/dev/cli.md)
- [Run artifacts and exact replay](docs/dev/run_artifacts_and_replay.md)
- [Changing studies](docs/dev/studies.md)
- [Testing and validation](docs/dev/testing.md)
- [Architecture health](docs/dev/architecture_health.md)

Public examples use `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`. Internal extension points, targeted tests, and exact commands are maintained in the linked guides.

When implementation and prose disagree, verify the current code/tests and correct the guide in the same bounded change. Build documentation with `mkdocs build --strict` and report any unavailable validation honestly.

```agent-docs
path: AGENTS.md
path: docs/dev/add_problem.md
path: docs/dev/add_operator.md
path: docs/dev/add_algorithm.md
path: docs/dev/add_backend.md
path: docs/dev/add_metric.md
path: docs/dev/cli.md
path: docs/dev/run_artifacts_and_replay.md
path: docs/dev/studies.md
path: docs/dev/testing.md
path: docs/dev/architecture_health.md
command: mkdocs build --strict
```

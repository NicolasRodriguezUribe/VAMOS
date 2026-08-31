# Scope

Applies only to `src/vamos/experiment/cli/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `main.py` owns top-level dispatch; focused command modules own their parsers and execution.
- Base optimization arguments are assembled through `args.py` and `args_*.py`; keep CLI and machine-readable spec defaults aligned through the existing spec-argument machinery.
- Parse and validate before executing. Help paths must remain side-effect free and usable without optional runtime services.
- CLI artifact commands delegate to the canonical artifact API: `results inspect`, `results verify`, and exact `reproduce` stay distinct.
- Preserve useful error messages and nonzero exit codes. Do not silently replace an unavailable algorithm, backend, problem, or file.

The current top-level subcommands are `quickstart`, `create-problem`, `summarize`, `open-results`, `results`, `reproduce`, `study`, `ablation`, `assist`, `check`, `bench`, `studio`, `zoo`, `tune`, and `profile`. The standard optimization path uses top-level options without a subcommand. The only current study subcommand is the read-only `study plan` preflight.

## Change route

Follow [Changing the CLI](/docs/dev/cli.md). Add dispatch, parser/help, implementation, subprocess tests, and user docs as one bounded change.

## Targeted validation

Run `python -m pytest -q tests/experiment/test_cli_consolidation.py tests/experiment/test_cli_config_validation.py tests/experiment/test_cli_run_artifacts.py` plus the test for the affected command.

```agent-docs
path: src/vamos/experiment/cli/main.py
path: src/vamos/experiment/cli/args.py
path: src/vamos/experiment/cli/run_artifact_cli.py
path: src/vamos/experiment/cli/study.py
path: tests/experiment/test_cli_consolidation.py
path: tests/experiment/test_cli_config_validation.py
path: tests/experiment/test_cli_run_artifacts.py
path: docs/dev/cli.md
cli: vamos --help
cli: vamos results inspect --help
cli: vamos results verify --help
cli: vamos reproduce --help
cli: vamos study plan --help
cli: vamos quickstart --help
cli: vamos create-problem --help
cli: vamos summarize --help
cli: vamos open-results --help
cli: vamos ablation --help
cli: vamos assist --help
cli: vamos check --help
cli: vamos bench --help
cli: vamos studio --help
cli: vamos zoo --help
cli: vamos tune --help
cli: vamos profile --help
command: python -m pytest -q tests/experiment/test_cli_consolidation.py tests/experiment/test_cli_config_validation.py tests/experiment/test_cli_run_artifacts.py
```

# Scope

Applies only to `src/vamos/experiment/study/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `types.py` owns `StudyTask` and the derived `StudyResult` summary; `runner.py` executes a supplied sequence through `StudyRunner.run(...)`.
- `api.py` is the supported orchestration route and supplies the current single-run service to `StudyRunner`.
- A study summary may reference or aggregate run results but must not create a second copy of canonical per-run arrays or specifications.
- `CSVPersister` exports a derived table only. Durable study state, resume/retry semantics, and a StudyManifest are not implemented here yet.
- Preserve task order, explicit seeds, indicator failure reporting, and optional-dependency behavior.

## Change route

Follow [Changing studies](/docs/dev/studies.md). Check run-artifact ownership before adding any persistence field or output.

## Targeted validation

Run `python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py` plus the nearest study/indicator tests.

```agent-docs
path: src/vamos/experiment/study/api.py
path: src/vamos/experiment/study/runner.py
path: src/vamos/experiment/study/types.py
path: src/vamos/experiment/study/persistence.py
path: tests/experiment/test_ablation_study_api.py
path: tests/experiment/test_cli_ablation.py
path: docs/dev/studies.md
symbol: vamos.experiment.study.runner:StudyRunner
symbol: vamos.experiment.study.api:run_study
command: python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```

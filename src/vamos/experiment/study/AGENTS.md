# Scope

Applies only to `src/vamos/experiment/study/**`.

Inherits all repository-wide rules from `/AGENTS.md`. This file contains local deltas only.

## Responsibility and invariants

- `models.py`, `planning.py`, `creation.py`, and `loading.py` own the canonical durable model, immutable plan, atomic creation, and data-only loaded view.
- `execution.py` owns the no-argument, newly-created-only sequential `Study.run()` path. It executes ascending `task_id`; `plan_index` is presentation metadata.
- `journal.py` validates and replays immutable events; `checkpoint_projection.py` validates lagging checkpoints and builds the effective immutable view without writes.
- `commits.py`, `run_publication.py`, and `writing.py` own event/checkpoint commits, verified RunManifest linkage, fixed failure safety, and atomic file primitives.
- `runner.py`, `types.py`, and `api.py` remain only for unmigrated in-memory benchmark/ablation callers. The durable runner never delegates to them and they are not another persisted authority.
- A study summary may reference or aggregate run results but must not create a second copy of canonical per-run arrays or specifications.
- RunManifest remains the sole authority for resolved per-run truth, environment, provenance, arrays, replayability, and outcome. Study attempts retain only bounded root-relative manifest references.
- `CSVPersister` exports a derived table only. It is not canonical state and has no compatibility route into StudyManifest.
- Resume, retry, selectable failure policy, cancellation, locks, leases, parallel workers, study CLI, migration, and summaries remain unimplemented; follow the ordered contract roadmap.
- Preserve task order, explicit seeds, indicator failure reporting, and optional-dependency behavior.

## Change route

Follow [Changing studies](/docs/dev/studies.md) and the linked StudyManifest contract and acceptance inventory. Check run-artifact ownership before adding any persistence field or output. Implement roadmap slices in order and do not add a legacy layout, compatibility reader, second runner, or automatic retry.

## Targeted validation

Run `python -m pytest -q tests/experiment/study_manifest tests/experiment/run_artifacts/test_verification_replay.py tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py` plus the nearest study/indicator tests.

```agent-docs
path: src/vamos/experiment/study/api.py
path: src/vamos/experiment/study/runner.py
path: src/vamos/experiment/study/types.py
path: src/vamos/experiment/study/persistence.py
path: src/vamos/experiment/study/models.py
path: src/vamos/experiment/study/planning.py
path: src/vamos/experiment/study/creation.py
path: src/vamos/experiment/study/loading.py
path: src/vamos/experiment/study/execution.py
path: src/vamos/experiment/study/journal.py
path: src/vamos/experiment/study/checkpoint_projection.py
path: src/vamos/experiment/study/run_publication.py
path: src/vamos/study_artifacts.py
path: tests/experiment/study_manifest
path: tests/experiment/test_ablation_study_api.py
path: tests/experiment/test_cli_ablation.py
path: docs/dev/studies.md
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
path: docs/dev/adr/0008-durable-study-manifest-contract.md
symbol: vamos.experiment.study.runner:StudyRunner
symbol: vamos.experiment.study.api:run_study
symbol: vamos.experiment.study.models:Study.run
symbol: vamos.study_artifacts:create_study
symbol: vamos.study_artifacts:load_study
command: python -m pytest -q tests/experiment/study_manifest tests/experiment/run_artifacts/test_verification_replay.py tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```

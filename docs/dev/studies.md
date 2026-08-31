# Creating and loading durable studies

The public StudyManifest v1 slice separates planning from execution. Creation
resolves every task and publishes a relocatable directory atomically; loading
verifies that directory using data-only readers.

```python
from vamos import StudySpec, create_study, load_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii"],
    seeds=[0, 1],
    max_evaluations=10_000,
)

study = create_study(spec, output="studies/example")
loaded = load_study("studies/example")
```

`create_study` does not optimize, create attempts, or publish runs. It freezes
current problem, algorithm, operator, backend, population, termination, budget,
and seed resolution into the immutable `plan.json`. Every task starts
`pending`. `load_study` checks closed schemas, canonical bytes, semantic and
file hashes, byte lengths, identities, counts, event head, resource limits, and
root-confined references without importing a component named by stored data.

The returned handle exposes immutable `study_id`, `plan_id`, `status`, `spec`,
`plan`, `tasks`, and `root` values. Equality includes the root path; after a
whole-directory move, compare persisted IDs and task IDs. Nested JSON values
are defensive immutable copies.

## Initial layout

A nonempty plan creates exactly:

```text
<study>/
├── study-manifest.json
├── study-spec.json
├── plan.json
├── events/
│   └── 00000000000000000001.json
└── tasks/
    └── <task-digest>/
        └── task.json
```

An empty plan omits `tasks/`. This slice creates no `attempts/`, `runs/`,
`coordination/`, or `derived/` path.

## Identities

- `study_id` is a random UUIDv4 for one persisted study instance.
- `plan_id` identifies the sorted scientific task set. Labels, metadata,
  output path, and display order do not affect it.
- `task_id` is the canonical RunManifest identity of the complete resolved run
  specification. A scientific change, including seed or budget, changes it.
- `attempt_id` will identify one future execution claim; creation emits none.
- `run_id` belongs to one future canonical RunManifest; creation emits none.

The plan is immutable. Change scientific configuration by creating a new study
at a new absent destination. Reopen an existing study with `load_study`; every
existing creation destination is a collision.

## Current limitations

This slice has no durable runner, `run`, `resume`, `retry`, cancellation, study
CLI, locks, leases, workers, or summaries. Those operations remain sequenced by
the [StudyManifest contract](study_manifest_contract.md) and its
[acceptance inventory](study_manifest_acceptance_tests.md).

The older internal `StudyRunner`/`StudyTask` path remains temporarily for
existing benchmark and ablation callers. It is in-memory execution and derived
CSV reporting only; it is not another durable layout or reader.

## Required validation

```bash
python -m pytest -q tests/experiment/study_manifest
python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
python -m pytest -q tests/experiment/run_artifacts
```

```agent-docs
path: src/vamos/study_artifacts.py
path: src/vamos/experiment/study/models.py
path: src/vamos/experiment/study/planning.py
path: src/vamos/experiment/study/creation.py
path: src/vamos/experiment/study/loading.py
path: tests/experiment/study_manifest
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
path: docs/dev/adr/0008-durable-study-manifest-contract.md
symbol: vamos.study_artifacts:StudySpec
symbol: vamos.study_artifacts:create_study
symbol: vamos.study_artifacts:load_study
command: python -m pytest -q tests/experiment/study_manifest
command: python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```

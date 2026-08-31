# Creating, running, and loading durable studies

StudyManifest v1 separates planning from execution. Creation resolves every
task and atomically publishes a relocatable directory. `Study.run()` executes a
newly created study sequentially. `Study.resume()` and `Study.retry()` perform
explicit single-process recovery. Loading verifies and derives the effective
state without executing components or repairing checkpoints.

```python
from vamos import StudySpec, create_study, load_study

spec = StudySpec(
    problems=["zdt1", "zdt2"],
    algorithms=["nsgaii"],
    seeds=[0, 1],
    max_evaluations=10_000,
    on_error="fail_fast",  # or "continue"; persisted before execution
)

created = create_study(spec, output="studies/example")
completed = created.run()
loaded = load_study("studies/example")  # data-only

# Later, after a paused/interrupted execution:
resumed = loaded.resume()  # pending and eligible interrupted tasks
retried = resumed.retry(failed_only=True)  # explicit failed-task consent
```

`create_study` performs no optimization. It freezes the problem, dimensions,
algorithm, typed configuration, operators, backend, evaluation strategy,
population, termination, budget, and concrete seed into immutable `plan.json`.
Every task starts `pending` and creation writes no attempt or run.

`Study.run()` accepts no policy or scheduler arguments. It reopens the persisted
study, requires the pristine `created` state, executes tasks in ascending
canonical `task_id` order, and returns a freshly loaded immutable handle. The
old handle remains a `created` snapshot. `plan_index` is display metadata and
does not determine execution order or scientific identity.

Before objective evaluation, the runner durably publishes the execution start,
one created/running `AttemptRecord`, the `task_claimed` and `attempt_started`
events, and the matching task/root checkpoints. It reconstructs only supported
built-ins from the persisted resolved specification; it does not consult
current convenience defaults or call public replay.

Each successful task publishes one canonical RunManifest directory, fully
verifies and reloads it, and only then appends `attempt_succeeded`. The attempt
stores a bounded root-relative manifest descriptor; it does not copy resolved
configuration, environment, provenance, or numerical arrays. A new `run_id` is
always distinct from the `attempt_id`.

## Canonical layouts

A newly created nonempty study contains:

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

After a successful task, the affected subtree also contains:

```text
<study>/
├── events/<20-digit-sequence>.json
├── tasks/<task-digest>/attempts/<attempt-id>.json
└── runs/<run-id>/
    ├── manifest.json
    ├── environment.json
    └── result.npz
```

An empty plan omits `tasks/`. Running it appends only the completion transition,
creates no attempt or run, and moves `created` directly to `completed`.
Execution writes no `coordination/`, `derived/`, CSV, or summary path.

## Journal and failure behavior

Events are immutable, gap-free, hash-chained canonical files. Root, task, and
nonterminal-attempt documents are mutable checkpoints. A valid event newer than
a checkpoint is authoritative: `load_study()` derives the effective immutable
view but leaves every byte unchanged. A checkpoint ahead of the journal, a gap,
a duplicate/forked event, an invalid transition, or an inconsistent run
reference is an integrity error.

The published `StudySpec.on_error` is the only execution policy. `run()` accepts
no policy override. Reconstruction and objective exceptions are task failures:
the runner first publishes and verifies a canonical failed run, then commits
the failed attempt and task. Under `fail_fast`, the study becomes `paused` and
later tasks remain `pending`. Under `continue`, later independent tasks run and
the final state is `completed_with_failures`. Both outcomes return a freshly
loaded immutable `Study`; failure details remain bounded and sanitized in the
attempt, task, and event records.

Journal, checkpoint, scheduler, publication, verification, integrity, and
atomic-storage failures are infrastructure failures. They stop both policies
immediately and raise a typed `StudyInfrastructureError`. When journal and
checkpoint authority remain trustworthy and writable, the root becomes
`failed` without inventing a task result. An inability to publish or verify a
terminal run leaves the attempt/task/study explicitly `running`, with any
complete run unreferenced, for later reconciliation.

## Cancellation

`Study.cancel()` durably cancels a `created` or `paused` study. Every unclaimed
task becomes `cancelled`, the study becomes `cancelled`, and the returned handle
is freshly loaded. If this process currently owns `Study.run()`, `cancel()`
records an in-memory cooperative request; the runner observes it at the next
safe boundary, cancels an active attempt without fabricating a RunManifest,
cancels all unclaimed tasks, and commits `study_cancelled`. The immediate return
from that request is the current running snapshot; the `run()` return carries
the terminal cancelled state.

`KeyboardInterrupt` during reconstruction or objective evaluation follows the
same durable cancellation protocol. A forced process death performs no later
write and leaves running work for the next explicit reconciliation operation.
This slice does not claim that another process can safely cancel a running
study.

## Reconciliation, resume, and retry

Every resume or retry begins with a fresh verified load and an explicit
reconciliation write phase. A valid journal head refreshes lagging attempt,
task, and root checkpoints. One prior running attempt is recovered only from
its claim event's reserved run UUID: the exact expected RunManifest must verify
and match the immutable task/resolved specification. A missing expected output
interrupts the attempt; a corrupt or mismatched expected output is refused.
Unrelated run directories remain unreferenced and never imply success.

`Study.resume()` runs pending tasks and interrupted tasks that remain below the
persisted attempt bound. It never reruns success and does not include failed
tasks unless `retry_failed=True` is explicit. `Study.retry(failed_only=True)`
selects retryable failed tasks; `failed_only=False` also selects interrupted
tasks. Each retry keeps the task ID and immutable earlier attempts, while using
a fresh attempt, execution, and run UUID. Objective-execution failures are
retryable; deterministic reconstruction/configuration and integrity failures
are not. The persisted default limit is three attempts.

Both methods return a freshly loaded immutable `Study`; the calling handle
remains a snapshot. When no eligible task exists, resume returns a fresh handle
without adding an event or changing a byte. Before any new claim, persisted
built-in configuration is reconstructed and prior canonical run evidence, when
present, must match the current material environment exactly. There is no
environment-change override in this slice.

## Identities and immutable view

- `study_id` is a random UUIDv4 for one persisted study instance.
- `plan_id` identifies the sorted scientific task set.
- `task_id` is the canonical RunManifest identity of the complete resolved run.
- `attempt_id` identifies one durable attempt.
- `run_id` identifies its separate canonical RunManifest execution.

The returned handle exposes immutable `study_id`, `plan_id`, `status`, `spec`,
`plan`, `tasks`, `attempts`, `events`, and `root`. Nested JSON values are
defensive immutable copies. Moving the complete directory preserves persisted
identities and root-relative references.

## Deliberate limits

There is no study CLI, parallelism, cross-process ownership guarantee, lock,
lease, heartbeat, worker, migration, summary, or CSV behavior in this slice.
Calling `run()` on `running`, `paused`, `completed`,
`completed_with_failures`, `failed`, or `cancelled` state is an actionable typed
error. Obvious same-process reentry is also rejected.

The older internal `StudyRunner`/`StudyTask` path remains temporarily for
existing benchmark and ablation callers. It is not a durable layout or reader,
and `Study.run()` never delegates to it.

## Required validation

```bash
python -m pytest -q tests/experiment/study_manifest
python -m pytest -q tests/experiment/run_artifacts
python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```

```agent-docs
path: src/vamos/study_artifacts.py
path: src/vamos/experiment/study/models.py
path: src/vamos/experiment/study/planning.py
path: src/vamos/experiment/study/creation.py
path: src/vamos/experiment/study/loading.py
path: src/vamos/experiment/study/execution.py
path: src/vamos/experiment/study/failure_policy.py
path: src/vamos/experiment/study/cancellation.py
path: src/vamos/experiment/study/journal.py
path: src/vamos/experiment/study/checkpoint_projection.py
path: src/vamos/experiment/artifacts/resolved_reconstruction.py
path: tests/experiment/study_manifest
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
path: docs/dev/adr/0008-durable-study-manifest-contract.md
symbol: vamos.study_artifacts:StudySpec
symbol: vamos.study_artifacts:create_study
symbol: vamos.study_artifacts:load_study
symbol: vamos.experiment.study.models:Study.run
command: python -m pytest -q tests/experiment/study_manifest
command: python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
```

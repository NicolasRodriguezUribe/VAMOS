# ADR 0008: Durable StudyManifest v1 contract

- Status: Accepted for pre-release implementation
- Date: 2026-08-31
- Decision owners: VAMOS maintainers
- Contract: [Durable study and StudyManifest v1](../study_manifest_contract.md)
- Acceptance specification: [StudyManifest v1 acceptance](../study_manifest_acceptance_tests.md)

## Context

VAMOS currently executes studies as an in-memory loop. Canonical RunManifest
directories make each completed run durable, but no study-level authority
records the immutable task set, attempts, interruption, partial completion,
resume, retry, or concurrent ownership. CSV summaries are derived exports and
cannot safely supply that authority.

Adding persistence incidentally to the existing runner would leave identity,
failure, recovery, and atomicity decisions to implementation accidents. The
pre-release project can instead establish one contract before replacing the
current path.

## Decision

Adopt `vamos.study-manifest` version `1.0.0` and its companion v1 documents as
specified by the linked contract. A study has an immutable resolved plan,
stable RunManifest-compatible task identities, append-only one-file events,
reconstructible root/task checkpoints, immutable terminal attempts, and only
root-relative references to canonical run directories.

Creation and execution are separate. Creation atomically publishes a complete
planned study into an absent destination. Execution, resume, and explicit retry
share one execution service. Successful runs are published and verified before
success state is committed. Fail-fast is the default; continue is explicit;
automatic retry does not exist. Data-only load, inspect, verification, and
summary paths cannot import plugins, execute code, contact a network, or escape
the study root.

V1 concurrency is limited to processes on one host and a local filesystem. A
short study lock, expiring task leases, monotonically increasing claim epochs,
and fencing tokens prevent stale publication. Distributed coordination is a
later provider boundary, not a second schema.

The public direction is `StudySpec`, `create_study`, `load_study`, and a thin
persisted `Study` handle, plus the `vamos study` command group. The current
`StudyTask`, `StudyRunner`, `run_study`, study-local `StudyResult`, and CSV-led
orchestration are replaced after all callers migrate; no legacy persisted
study reader or dual runner remains.

### Implemented bounded slices

Atomic create/data-only load, the sequential durable runner, persisted
failure/cancellation policy, explicit reconciliation/resume/retry, and
read-only planning preflight are implemented.
`Study.run()` accepts a pristine `created` study, executes ascending `task_id`,
reconstructs the frozen built-in resolved spec, durably starts one attempt, and
publishes a fully verified canonical RunManifest before success. Valid newer
journal events are authoritative during data-only load; loading derives an
effective view and does not repair checkpoints.

The sequential slice still has no state-mutating CLI, locks, leases, workers, or cross-process
guarantee. Its published `on_error` policy is authoritative:
a verified task failure pauses fail-fast execution or is retained while
continue execution advances to `completed_with_failures`. Infrastructure
failure stops either policy and never becomes a task outcome. `Study.cancel()`
handles idle cancellation and same-process cooperative requests;
`KeyboardInterrupt` follows the same durable cancellation protocol. Explicit
reconciliation settles an interrupted single-owner attempt before resume or
retry claims fresh work. Retry remains explicit and bounded, successful tasks
never rerun, and the future environment-override and coordination Goals remain
required to realize those parts of this ADR.

## Consequences

- Study state can be audited, relocated, reconciled, and resumed without
  rerunning successful tasks or interpreting filenames.
- RunManifest remains the sole owner of per-run resolved truth and numerical
  output; study documents hold verified references, not copies.
- More files and explicit commits are required, but a transition touches only
  one event, affected entity checkpoints, and the small root checkpoint.
- Existing output destinations become strict collision boundaries.
- Invalid state, corruption, environment incompatibility, task failure, and
  infrastructure failure remain distinguishable in Python and CLI results.
- Pre-release migration intentionally deletes superseded paths rather than
  preserving aliases, importers, or competing semantics.

## Alternatives rejected

- A summary CSV as durable state: lossy, non-atomic, and unable to represent
  attempts, claims, or recovery.
- One mutable monolithic manifest: rewrites the task universe and supplies no
  durable audit boundary.
- Shared JSONL events: partial append and cross-platform locking behavior are
  harder to make unambiguous than one atomically published event per file.
- Rerun-on-resume: duplicates work and can hide missing or corrupt success.
- Automatic retry: obscures attempt history and changes resource use without
  an explicit user decision.
- Absolute run paths: break relocation and widen the data-only attack surface.
- A compatibility facade over the current runner: creates two authorities and
  postpones caller migration.

## Validation

The linked SA-001 through SA-074 specification is normative. Ten sanitized
machine-readable examples exercise representative valid and invalid states.
Active repository tests validate their hashes, paths, transitions, documented
errors, coverage, and links. Implementation Goals must turn the applicable SA
rows into executable production tests without weakening or marking them xfail.

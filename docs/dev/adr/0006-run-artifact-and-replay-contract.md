# ADR 0006: Run artifact and replay contract

## Status

Accepted

## Context

VAMOS currently serializes overlapping run information through several
independent paths. The CLI writes rich `metadata.json`, a flat
`resolved_config.json`, CSV arrays, timing, and an environment lock. The public
Python `save_result` helper writes CSV arrays and only solution/objective counts.
`OptimizationResult.meta`, study rows, analysis discovery, and Studio loaders
use still other shapes.

A current CLI-emitted `resolved_config.json` cannot be passed back to the same
CLI because `--config` accepts the hierarchical `ExperimentSpec` v1 shape. The
published replay recipe also expects `pop_size` while CLI storage emits
`population_size`. There is no public symmetric loader or replay operation.
Fixing either key or parser in isolation would preserve the conflicting sources
of truth.

The project needs a durable boundary that preserves user intent, actual
resolved execution choices, numerical results, status, provenance, integrity,
and an honest replay guarantee without making normal inspection execute code.

## Decision

Adopt the normative contract in
[`docs/dev/run_artifact_contract.md`](../run_artifact_contract.md):

1. `ExperimentSpec` remains user intent. It is retained independently and is
   not represented as if every runtime choice were already known.
2. `ResolvedRunSpec` records every effective deterministic execution choice
   after defaults, aliases, components, operators, reference directions,
   termination, seed, and backend/evaluation strategy are resolved.
3. `RunManifest` is the single versioned envelope and index for one execution
   attempt. Its identity is `vamos.run-manifest`, initial schema version
   `1.0.0`.
4. The manifest embeds both requested and resolved specs because they are compact
   and mutually constrained. A separate canonical `resolved_spec.json` would
   introduce an unnecessary consistency boundary.
5. Numerical arrays are stored outside the manifest in a safe `result.npz`
   `ResultBundle`, loaded with `allow_pickle=False`; object dtypes are forbidden.
6. A full environment/package snapshot is a hashed external JSON artifact.
   Optional metrics/events and human-readable CSV files are referenced by role.
7. `manifest.json` and `result.npz` are authoritative. `FUN.csv`, `X.csv`, and
   `G.csv` are non-authoritative compatibility views.
8. Artifact paths are relative and confined to the run directory. All referenced
   files carry SHA-256 and byte length. The manifest carries a canonical
   self-hash.
9. Status is one of `running`, `succeeded`, `failed`, `partial`, or `cancelled`.
   A terminal manifest is committed atomically after its referenced artifacts.
10. Loading is data-only: no arbitrary imports, dynamic code execution, pickle,
    network, shell commands, or traversal outside the run directory. Custom-code
    resolution is reserved for explicit trusted replay.
11. Replayability uses `exact`, `compatible`, `best_effort`, `manual`, and
    `unavailable`. Exact replay is limited to a deterministic path with identical
    resolved spec, implementation, backend, and materially equivalent
    environment. Overrides are explicit and can only downgrade the guarantee.
12. Use a stable content-derived `task_id` plus a unique UUIDv4 `run_id` for each
    execution attempt. This distinguishes repeated configurations, retries, and
    provenance changes.
13. Current layouts are explicitly `legacy-cli-v0` and
    `legacy-python-save-v0`. Readers recognize them, preserve unknown/missing
    evidence, and never fabricate provenance. Migration is in-memory unless the
    user requests a new destination.
14. The future Python facade combines simple `save_result`/`load_result` helpers
    with an advanced `load_run` object. `reproduce` is the only operation that
    executes optimization. The CLI provides `results inspect` and
    `reproduce [--verify-only]`; `--config` remains solely for ExperimentSpec.
15. A future `StudyManifest` references stable task/run IDs and manifest paths;
    it does not duplicate run specifications or arrays.

## Compatibility constraints

- Existing `save_result(result, path)` call sites remain valid; the implementation
  may add a useful return value and upgrade the written layout.
- `vamos.ux.api.save_result` remains an import-compatible alias while the new
  functions become top-level facade exports.
- Current CLI/Python run directories remain readable through explicit legacy
  detectors for at least two minor releases after v1 becomes the default.
- Legacy provenance gaps remain visible and prevent an exact replay claim.
- CSV compatibility exports remain available during the migration window.
- Unknown future major schema versions are rejected, not guessed.
- Existing artifacts are never rewritten by ordinary load, verify, or in-memory
  migration.

## Alternatives considered

### Reuse `ExperimentSpec` as the output

Rejected. Requested intent and resolved execution have different ownership and
truth conditions. Conflating them recreates the current CLI round-trip failure
and loses the explanation of which defaults were applied.

### Make the current flat `resolved_config.json` acceptable input

Rejected as the architectural solution. It lacks complete algorithm/operator
configuration, result identity, status, integrity, environment linkage, failure
semantics, and a safe public loader. A legacy mapper may read it, but it is not
the v1 contract.

### Put arrays directly in the manifest

Rejected. Large numerical data would make normal inspection expensive, impair
dtype/shape fidelity, complicate resource limits, and turn the manifest into a
god object.

### Use CSV as the canonical result

Rejected. CSV does not reliably preserve dtype, byte order, empty shapes,
aligned optional arrays, or nested population/archive roles. CSV remains useful
as a human-readable view.

### Use pickle or cloudpickle for custom components/results

Rejected. Normal loading could execute arbitrary code, the representation is
Python/version fragile, and missing custom code would prevent safe result
inspection.

### Use only an object method API

Rejected as the sole public design. `OptimizationResult.load()` would blur the
difference between a result and a complete stored run. Top-level helpers serve
beginners; `StoredRun` exposes advanced manifest/provenance/compatibility data.

### Use a configuration hash as the only run identity

Rejected. It cannot distinguish intentional repetitions, retries, or attempts
under different source/environment provenance. The contract uses both task and
attempt identity.

### Promise exact replay across backends or materially different environments

Rejected. Current backends may be deterministic within themselves without
being bitwise identical to one another. The contract reports compatibility and
requires explicit acceptance of downgraded guarantees.

## Consequences

### Positive

- Python, CLI, studies, analysis, reporting, and migration share one source of
  truth.
- Stored results remain inspectable even when custom code or plugins are absent.
- Exact replay claims become testable rather than inferred from a seed alone.
- Current artifacts can be migrated without rewriting or inventing evidence.
- Atomic status transitions make failed/interrupted runs distinguishable from
  corrupt successful runs.
- Future study resume/retry work can reference run attempts without redesigning
  the run artifact.

### Costs and risks

- Writers and readers must implement canonicalization, hashing, path
  confinement, resource limits, atomic replacement, and legacy mapping.
- A manifest plus NPZ/environment files is more structured than the current
  three-file Python save helper.
- Maintaining an honest compatibility matrix requires versioned component and
  backend policies.
- Exact replay of dirty or custom source is deliberately uncommon unless
  stronger source capture is explicitly enabled.
- SHA-256 detects accidental corruption but does not authenticate a malicious
  artifact without a future signature extension.

## Relationship to future StudyManifest

The run contract is complete without a study. A future StudyManifest owns task
planning and attempt/status references. It links `study_id` and stable
`task_id` values to one or more `run_id`/relative manifest paths, enabling
resume, retry, partial failure, and aggregation. Run manifests may retain a
reverse study link, but neither side duplicates the other's resolved spec or
numerical bundle.

## Implementation boundary

This ADR freezes design only. The implementation sequence begins with data
models/validation, legacy recognition, canonical Python save/load, and one
built-in round trip. CLI execution replay, StorageObserver integration,
analysis migration, durable studies, and custom-code execution are later,
separate changes.

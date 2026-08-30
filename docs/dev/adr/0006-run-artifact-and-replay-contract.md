# ADR 0006: Canonical v1 run-artifact contract

## Status

Accepted and implemented, including exact built-in replay

## Context

Pre-release VAMOS had multiple persistence prototypes. Python helpers, CLI
observers, studies, analysis, and Studio each selected different subsets of
configuration, arrays, timing, and environment state. Those outputs could not
be loaded symmetrically, did not preserve requested versus effective choices,
and encouraged consumers to infer facts from filenames or directory names.

Because no stable public release contract exists, preserving those prototypes
would create the migration burden it attempts to avoid. The project needs one
safe artifact boundary before release.

## Decision

Adopt the normative contract in
[`docs/dev/run_artifact_contract.md`](../run_artifact_contract.md):

1. One schema is supported: document identity `vamos.run-manifest`, version
   `1.0.0`.
2. A succeeded run contains `manifest.json`, `result.npz`, and
   `environment.json`. No parallel persistence views are emitted.
3. `ExperimentSpec` records requested intent. `ResolvedRunSpec` records actual
   execution choices. Both are embedded in the manifest and the resolved spec
   determines `task_id`.
4. A requested seed may be null, but the seed is resolved to an actual integer
   before stochastic execution and persisted in the resolved spec. Zero is a
   valid seed; unknown values have no sentinel.
5. The result NPZ is data-only, uses `allow_pickle=False`, preserves numerical
   dtype/shape/value information, and is inspected under finite resource limits
   before allocation.
6. Artifact paths are relative and confined. Referenced bytes have length and
   SHA-256 evidence, and the terminal manifest has a canonical semantic
   self-hash.
7. Writes are non-destructive and atomic at directory publication. Existing
   destinations always collide.
8. `vamos.save_result`, CLI `StorageObserver`, and every supported run path use
   the same writer. Analysis and Studio use the same readers. Studies reference
   runs and never mirror or reconstruct their files.
9. The only public saver is top-level `vamos.save_result`. A manually created
   result requires explicit complete requested and resolved specs; otherwise a
   typed `IncompleteRunMetadataError` is raised.
10. Loading, inspection, and verification are inert data access. They never
    execute optimization, resolve plugins, import manifest-provided code, use
    shell/network, or mutate artifacts.
11. Earlier pre-release output layouts are unsupported. There are no readers,
    detectors, adapters, migrations, aliases, warnings, or dual-format writers.
    The actionable response is to regenerate with the current VAMOS version.
12. A future StudyManifest may reference run/task identities but will not
    duplicate run specifications or arrays.
13. Exact replay is a separate explicit operation for verified built-in
    components in the same material environment and backend. It reconstructs
    solely from the persisted resolved spec and refuses rather than downgrade.
14. An executed replay creates a new canonical run with source/root lineage,
    plan/source hashes, and bitwise F/X plus auxiliary-array comparison.
15. There is no `reproduce --verify-only`; verification is `results verify`.

## Consequences

Positive consequences:

- one ownership path prevents divergent metadata and array truth;
- Python and CLI outputs are symmetric and relocatable;
- analysis no longer depends on directory heuristics;
- provenance gaps and manual context are explicit rather than fabricated;
- loading untrusted data has a narrow, bounded, non-executable surface;
- pre-release cleanup avoids permanent compatibility machinery.

Costs and risks:

- all writers and consumers must change together;
- callers with manually constructed results must provide complete execution
  context;
- an existing destination must be deliberately changed rather than overwritten;
- SHA-256 provides integrity detection but not authentication;
- exact replay refuses when source or environment identity is weak;
- custom/plugin/cross-backend replay remains intentionally unsupported.

## Alternatives considered

### Keep multiple outputs and declare one authoritative

Rejected. Duplicate files still require synchronization, expand verification,
and invite consumers to select a convenient stale view.

### Accept earlier formats through a registry or fallback parser

Rejected. This would freeze incomplete prototypes before the first stable
release and retain fabricated or missing metadata semantics.

### Use requested configuration as the resolved record

Rejected. It loses defaults, provider resolution, actual backend/operator
choices, and the generated seed.

### Infer manual metadata from array shapes

Rejected. Shapes cannot establish algorithm, termination, backend, seed,
provenance, or determinism.

### Store arrays as JSON or executable Python serialization

Rejected. JSON loses efficient numerical fidelity; pickle-like formats make
ordinary loading executable and version-fragile.

### Implement reproduction in this decision

Rejected. Data persistence and safe loading are complete independently. Trusted
component resolution, environment comparison, override policy, and execution
belong to a future explicit reproduction design.

## Future boundary

Custom/plugin reconstruction, cross-backend or best-effort execution, automatic
environment installation, a durable StudyManifest, and study resume/retry are
not part of this implementation. They must build on this schema without adding
alternate run persistence paths.

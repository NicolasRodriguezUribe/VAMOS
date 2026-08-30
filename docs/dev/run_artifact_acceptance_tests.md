# Canonical v1 run-artifact acceptance specification

This matrix is normative for schema `vamos.run-manifest` `1.0.0`. IDs are
contiguous and describe observable behavior, not implementation details.

| ID | Acceptance behavior | Required evidence |
|---|---|---|
| RA-001 | Python built-in round trip | Optimize, save, load; arrays, manifest identity, task/run IDs, and hashes validate. |
| RA-002 | Rich resolved configuration | Algorithm, operator parameters, backend, evaluation, termination, population, defaults, and problem details survive. |
| RA-003 | Canonical directory only | A successful save contains exactly `manifest.json`, `result.npz`, and `environment.json`. |
| RA-004 | Numerical fidelity | Empty and nonempty arrays preserve values, shape, dtype width, byte order, boolean/integer/floating kind, and namespaces. |
| RA-005 | Outcome consistency | Counters, dimensions, result mode, runtime, termination, interruption, and metrics agree with arrays/execution. |
| RA-006 | Seed zero | Requested, exposed, resolved, and persisted seed remains integer zero. |
| RA-007 | Generated seed | Requested seed null is preserved; an integer is generated before execution, exposed, and persisted. |
| RA-008 | Manual context rejection | Manual result without both complete specs, or with only one, raises `IncompleteRunMetadataError` without publishing. |
| RA-009 | Manual context success | Manual result with explicit requested and resolved specs saves, loads, and is marked manual. |
| RA-010 | CLI convergence | CLI execution uses the canonical writer, records CLI provenance, and produces the same three-file layout. |
| RA-011 | Hook/archive integration | Hook traces and archive summaries are canonical manifest metrics; no independent run files are written. |
| RA-012 | Analysis convergence | Discovery scans manifests; array loading uses the canonical reader; aggregation uses resolved/outcome fields and handles absent metrics. |
| RA-013 | Studio convergence | Studio discovers manifests and loads result/archive arrays without filename or directory inference. |
| RA-014 | Study ownership | Study summaries may reference a run path but neither copy nor reconstruct run artifacts. |
| RA-015 | Public API | Saver/loader/model/error exports are top-level; no persistence export exists in `vamos.ux.api`; signatures match the contract. |
| RA-016 | Relocation and privacy | Moving the complete directory preserves loading; stored JSON has no source/destination personal path, hostname, account, or secret. |
| RA-017 | Integrity failures | Missing, length-modified, hash-modified, duplicate-key, malformed, and semantically inconsistent artifacts raise actionable typed errors. |
| RA-018 | Path confinement | Absolute, traversal, backslash, URI/drive, empty-segment, and available symlink escapes are rejected before external access. |
| RA-019 | Resource limits | Descriptor, ZIP/member, header, dtype, shape, size, element, depth, and compression-ratio limits reject before unsafe materialization. |
| RA-020 | Data-only loading | Load never calls optimize, imports recorded components, resolves plugins, executes code, uses pickle, invokes shell/network, or opens inert unknown roles. |
| RA-021 | Atomic non-destructive write | All existing destinations collide unchanged; injected failures publish nothing and clean only owned staging/lock state. |
| RA-022 | Status behavior | Succeeded runs require result/environment; failed runs remain inspectable and reject numerical-result access; terminal timestamps/integrity are required. |
| RA-023 | Unsupported format behavior | Missing/wrong manifest identity or version is uniformly rejected with a pre-release regeneration action and no filename classification. |
| RA-024 | Distribution and documentation | Clean wheel install exposes the same API and round trip; examples verify; docs build; prohibited active references and duplicate writers are absent. |
| RA-025 | Manifest-only inspection | Human/JSON inspection covers success, failure, and replay lineage without materializing arrays or executing/resolving code. |
| RA-026 | Full inert verification | All descriptors, hashes/lengths, paths, environment JSON, and NPZ structure are verified with independent structured dimensions and no execution. |
| RA-027 | Material environment policy | VAMOS/source fingerprint, Python, OS/architecture, NumPy/SciPy, backend/package/capabilities, BLAS, and thread evidence determine compatibility; missing evidence blocks exact. |
| RA-028 | Exact built-in reconstruction | All nine built-in algorithms reconstruct typed resolved configuration, operators, problem, termination, backend, and persisted seed with semantic equality before execution. |
| RA-029 | Bitwise comparison | F/X and deterministic auxiliary roles compare dtype, shape, logical order, and raw logical bytes; useful hashes/index/difference evidence accompanies mismatch. |
| RA-030 | Replay lineage and failures | Replay publishes a new immutable schema-1 attempt with source/root lineage and comparison; begun failures remain inspectable; refusal publishes nothing. |
| RA-031 | Replay API, CLI, and trust boundary | Top-level verify/reproduce and `results inspect/verify`/`reproduce` share services, stable JSON/exit codes, reject custom/plugins, and work from a clean wheel. |

## Test ownership

- `tests/experiment/run_artifacts/` owns RA-001–RA-009 and RA-015–RA-023.
- CLI/integration tests own RA-010–RA-011 and the CLI portion of RA-024.
- UX analysis/Studio tests own RA-012–RA-013.
- Study tests own RA-014.
- packaging, API snapshot, documentation, architecture, and repository absence
  checks own RA-024.
- `tests/experiment/run_artifacts/test_verification_replay.py` and
  `test_replay_matrix.py` own RA-025–RA-030.
- CLI, public API, security, and clean-wheel tests jointly own RA-031.

Linux CI should execute the symlink variants in RA-018. Platforms without
symlink privileges may skip only those capability-dependent cases; all lexical
confinement cases remain mandatory everywhere.

Custom/plugin/cross-backend/best-effort replay, earlier-format migration, and a
durable StudyManifest remain outside this matrix.

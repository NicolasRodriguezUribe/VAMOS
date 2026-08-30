# Run artifact acceptance-test specification

Status: **normative implementation acceptance plan**

Contract: [Run artifact, persistence, and replay contract](run_artifact_contract.md)

This document specifies future tests; it does not add skipped or `xfail`
placeholders. A production implementation is incomplete until its unlocked
rows pass. IDs are stable and MUST be cited by implementation pull requests.

## Test organization and shared rules

Proposed modules:

```text
tests/experiment/run_artifacts/
├── conftest.py
├── test_round_trip.py
├── test_replay.py
├── test_legacy.py
├── test_integrity.py
├── test_custom_components.py
└── test_public_api_cli.py
```

Test fixtures use tiny deterministic budgets, fixed clocks, fixed UUIDs, and
temporary directories. They import public functions for UX assertions and may
use internal data-model helpers for unit-level validation. Subprocess tests set
`PYTHONPATH` or install the test wheel explicitly so they cannot accidentally
exercise another checkout.

Every CLI assertion captures stdout, stderr, and exit status. Human output MUST
contain the named phrases below; JSON output MUST contain equivalent structured
fields: `operation`, `category`, `artifact_role` or `field`, `reason`,
`expected`, and `action` where applicable. Path/hash errors also contain
`path`, `expected_sha256`, `actual_sha256` or `state=missing`.

Canonical writer tests MUST parse every JSON document with duplicate-key
detection, open NPZ files with `allow_pickle=False`, recompute every descriptor
length/hash and the manifest self-hash, and assert that no absolute fixture path
appears in stored JSON.

## Shared fixtures

| Fixture | Definition |
|---|---|
| `minimal_nsgaii_case` | ZDT1, NSGA-II, NumPy, pop 8, 24 evaluations, seed 7, deterministic serial evaluation |
| `rich_nsgaii_case` | ZDT1, explicit SBX, polynomial mutation, tournament selection, clip repair, pop/offspring 8, seed 11 |
| `moead_case` | DTLZ2 with 3 objectives, MOEA/D, explicit lattice/reference directions, NumPy, seed 13 |
| `constrained_case` | Tiny built-in/test problem returning aligned `F`, `X`, `G`, and `CV` with `g_lte_0` convention |
| `legacy_cli_v0_dir` | Frozen current CLI output with rich metadata, flat resolved config, CSV arrays, timing, and lock |
| `legacy_python_v0_dir` | Frozen current `save_result` output with CSV arrays and count-only metadata |
| `fixed_provenance` | Clean fixed implementation/environment fingerprint, UTC timestamps, redacted command, deterministic backend |
| `cli_runner` | Subprocess helper returning argv, stdout, stderr, and status without shell interpolation |

## Core round trip — 8 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-001 | `test_round_trip.py::test_minimal_builtin_nsgaii_round_trip` | v1 models/writer/reader and Python facade exist | `minimal_nsgaii_case`, `fixed_provenance` | Optimize, `save_result`, move nothing, `load_run`, `load_result` | `manifest.json`, `result.npz`, `environment.json`; optional CSVs hash-valid | Loaded `F/X` have identical shape/dtype/values; manifest algorithm/problem/seed/backend match | N/A | N/A | One public save/load path preserves a minimal run | Count-only Python persistence |
| RA-002 | `test_round_trip.py::test_rich_nsgaii_resolved_operators_round_trip` | Config serializers implement resolved operator objects | `rich_nsgaii_case` | Save/load a run with explicit crossover, mutation, selection, repair and offspring size | Core files; manifest embeds full requested/resolved specs | Each operator ID and numeric parameter, result mode, pop/offspring sizes and default source compare equal | N/A | N/A | Rich typed config is not flattened or dropped | Operator/default loss during save |
| RA-003 | `test_round_trip.py::test_moead_reference_direction_contract_round_trip` | MOEA/D resolver emits lattice/direction evidence | `moead_case` | Save/load and recompute `task_id` | Core files; `result.npz` contains `reference_directions` when generated values are material | Aggregation, neighbors, replacement limit, partitions/count and directions compare exactly; task hash validates | N/A | N/A | MOEA/D compatibility population and directions are reproducible | A nominal config that cannot rebuild weight vectors |
| RA-004 | `test_round_trip.py::test_result_arrays_preserve_shape_dtype_and_values` | Safe NPZ reader/writer exists | Parameterized float32/float64 `F`, integer/float `X`, empty `(0,n)` arrays | Write/read bundle with no optimization | Core files | `np.array_equal`; exact dtype strings, byte order, two-dimensional empty shapes preserved | N/A | N/A | Numerical bundle is lossless, including empty arrays | CSV dtype/shape coercion |
| RA-005 | `test_round_trip.py::test_constraint_arrays_and_violation_semantics_round_trip` | Constraint convention is modeled | `constrained_case` | Save/load `F/X/G/CV`; validate row alignment and convention | `result.npz` with `G/CV`; manifest constraint convention | All arrays exact; `G <= 0` feasibility and stored/derived CV agree | N/A | N/A | Constraint meaning survives rather than merely bytes | Silent sign/convention changes |
| RA-006 | `test_round_trip.py::test_outcome_metrics_counters_and_termination_round_trip` | Outcome model exists | Minimal run with known evaluations, generations, monotonic runtime, termination reason, scalar metrics | Save/load and inspect outcome | Core files; optional `metrics.json` if extended metrics used | Evaluations/generations/runtime/termination/metrics equal; derived counts match arrays | `results inspect` later shows the same counters | 0 when CLI available | Outcome is not inferred from requested budget | Actual-vs-requested counter confusion |
| RA-007 | `test_round_trip.py::test_resolved_defaults_and_sources_survive_round_trip` | Resolver reports every applied default | Minimal run with algorithm/engine/pop/budget defaults intentionally omitted in intent | Save/load; compare requested vs resolved | Core files | Requested omissions remain omissions; resolved values and `defaults_applied` pointers/reasons present | Inspect labels requested vs resolved choices | 0 when CLI available | User intent remains distinct from effective state | False claim that input contained defaults |
| RA-008 | `test_round_trip.py::test_unknown_optional_fields_are_preserved_without_execution` | Supported-major parser retains unknowns | v1.1-like fixture with namespaced unknown fields and unknown artifact role pointing to inert file | Load, explicit migrate to new directory, reload | Original unchanged; migrated core files plus inert artifact | Unknown fields/descriptor retained byte-semantically; unknown role unopened (sentinel hook proves) | Inspect reports unknown retained extension, not failure | 0 | Forward-compatible optional data survives safely | Destructive read/write migration or eager unknown-role access |

## Replay — 6 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-009 | `test_replay.py::test_exact_replay_same_environment_matches_f_and_x` | Explicit replay implemented for built-ins | Saved `minimal_nsgaii_case` and identical fixed environment | `reproduce(..., accept_level="exact", output=...)` | Original unchanged; new complete run directory | `ReplayReport.level == "exact"`; original/replay `F/X` shape/dtype/value identical | Contains `Replay level: exact`, source/new run IDs and output | 0 | Exact means bitwise arrays for deterministic fixture | Seed-only or approximate “exact” claim |
| RA-010 | `test_replay.py::test_different_backend_requires_override_and_downgrades` | Compatibility policy and at least two test backend descriptors exist | Exact NumPy run; current/requested alternate backend | First replay without override; then with `backend=...`, `accept_level="compatible"` | No output on refusal; new derived run on accepted override | First raises structured compatibility error; second report is at most compatible and gets new task ID | Refusal names recorded/current backend and exact `--accept-level compatible --backend ...` action; accepted run warns non-exact | 5 then 0 | Backend never changes silently or retains exact/task identity | Silent backend substitution |
| RA-011 | `test_replay.py::test_environment_mismatch_returns_structured_report` | Environment comparator exists | Original/current differ in Python patch, NumPy, BLAS, threads | `load_run(...).compatibility()` and JSON verify-only | No new result directory | Report lists every original/current value, severity, declared/effective level and reasons | JSON contains `effective_level=compatible` and four differences | 0; 5 with `--require-level exact` | Integrity success and compatibility mismatch are distinct | Single vague “environment differs” warning |
| RA-012 | `test_replay.py::test_dirty_source_is_recorded_and_limits_exact_replay` | Source collector supports clean/dirty/unknown | Dirty Git fixture with deterministic normalized diff hash | Save, inspect, attempt exact replay without captured source snapshot | Core files contain `dirty=true`, `diff_sha256`; no diff content | Declared/effective ceiling compatible; provenance exposes hash but not source content/path secrets | Names dirty source as reason and safe action; no exact claim | 5 for exact requirement | Dirty state cannot masquerade as clean/exact | Missing dirty-tree provenance |
| RA-013 | `test_replay.py::test_existing_replay_output_directory_is_never_overwritten` | Replay output planner exists | Exact run plus existing directory with sentinel | Replay to occupied path | Existing sentinel unchanged; no partial replay files | Raises `OutputCollisionError` with candidate new path | Contains operation, path, `already exists`, `choose another output directory` | 8 | Replay is non-destructive | Accidental overwrite of scientific results |
| RA-014 | `test_replay.py::test_replay_never_substitutes_missing_component` | Component resolver and stable IDs exist | Manifest names unavailable algorithm/operator/backend; similarly named alternatives registered | Verify then reproduce | Original unchanged; no output attempt | Load succeeds; reproduce raises exact missing component IDs and no registry alternative invoked | Contains missing ID/provider and install/restore action; no “using fallback” | 6 | Replay matches stable identities or refuses | Silent algorithm/operator/backend fallback |

## Legacy compatibility — 6 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-015 | `test_legacy.py::test_current_cli_directory_is_recognized_read_only` | Legacy detector exists | `legacy_cli_v0_dir` frozen from current CLI | `load_run` and inspect without migration | Original six files unchanged; no manifest created | Layout=`legacy-cli-v0`; arrays and known metadata load; warnings list absent hashes/dirty state | Shows `Legacy layout: legacy-cli-v0`, missing guarantees and migrate action | 0 | Current runs remain inspectable without mutation | Breaking existing archives |
| RA-016 | `test_legacy.py::test_flat_resolved_config_maps_deterministically` | Legacy CLI mapping table implemented | Legacy flat config with selected NSGA-II variation and unrelated empty algorithm blocks | Map in memory twice and compare canonical resolved spec/task hash | No files written | `population_size` maps to pop size; selected variation maps; unrelated empty blocks discarded with note; same input gives same hash | `results inspect --json` exposes mapping decisions | 0 | Legacy mapping is deterministic, not a permissive parser | One-off `--config` aliasing and ambiguous keys |
| RA-017 | `test_legacy.py::test_count_only_python_save_is_recognized_or_actionably_rejected` | Python legacy detector exists | `legacy_python_v0_dir` | `load_result`; request reproduce; explicit migration attempt without supplemental spec | Original unchanged | Arrays load; layout correct; replay unavailable; migration reports exactly which spec/provenance is missing | Inspect names count-only layout; reproduce suggests loading data or supplying verified migration inputs | 0 inspect; 6 reproduce | Stored arrays remain useful without fabricated replay | Treating count metadata as complete run |
| RA-018 | `test_legacy.py::test_legacy_reader_never_fabricates_provenance` | Legacy mapper models unknown values | Legacy fixture missing Git, dirty state, start time, backend capabilities and package lock | Load and inspect compatibility | No files written | Fields are null/unknown with reasons; replay level below exact; no defaults inserted | Output enumerates missing provenance | 0 inspect; 5 exact requirement | Absence is preserved as evidence | Fake seed/backend/clean-tree claims |
| RA-019 | `test_legacy.py::test_future_major_schema_is_rejected_before_artifact_access` | Schema dispatcher exists | `document_type=vamos.run-manifest`, version `2.0.0`, artifact points to sentinel that fails if opened | `load_run`, CLI inspect | No writes; sentinel unopened | Raises `UnsupportedSchemaError` with supported major and migration guidance | Contains received `2.0.0`, supported `1.x`, upgrade action | 4 | Unsupported semantics are never guessed or touched | Unsafe fallback to legacy/known filenames |
| RA-020 | `test_legacy.py::test_known_old_schema_migrates_in_memory_and_only_writes_on_request` | At least one synthetic `1.0.0-pre`/known migration fixture exists | Known older manifest fixture and read-only original bytes | Load; explicit migrate to sibling; compare originals | Ordinary load creates nothing; explicit destination has v1 files; original hashes unchanged | In-memory model is current and preserves unknowns; migration idempotent | Inspect reports in-memory migration; migrate command names new path | 0 | Migration is explicit, non-mutating and deterministic | Silent in-place schema rewrite |

## Integrity and relocation — 7 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-021 | `test_integrity.py::test_run_directory_can_move_and_load` | Relative-path resolver exists | Complete minimal run saved under path A | Move entire directory to unrelated path B; load/verify all | Same relative contents at B; no references to A | Arrays/manifest IDs unchanged; all hashes pass; no original absolute path required | Verify reports intact at B | 0 | Run directory is relocatable | Persisted absolute-path dependency |
| RA-022 | `test_integrity.py::test_missing_artifact_error_identifies_role_path_hash_and_action` | Integrity errors structured | Complete run with `result.npz` removed after manifest creation | Load result and verify | Manifest/environment remain; result absent | `ArtifactIntegrityError` fields role=`result_bundle`, path, expected hash/bytes, state=`missing`, restore action | Same fields in human/JSON output | 3 | Missing canonical data never falls back to CSV | Vague file-not-found or silent repair |
| RA-023 | `test_integrity.py::test_modified_array_file_fails_hash_verification` | SHA-256 verification implemented | Complete run; flip one byte in `result.npz` without changing length | `load_result`, verify-only | Corrupt file preserved for evidence; no rewritten hashes | Error contains expected/actual SHA; NPZ not trusted | Contains `hash mismatch`, role/path, restore action | 3 | Exact bytes are integrity-protected | Analysis of modified arrays as authentic |
| RA-024 | `test_integrity.py::test_modified_manifest_fails_self_hash_or_schema_validation` | Canonical manifest self-hash implemented | Parameterize semantic valid field edit without hash update, malformed JSON, duplicate key | Inspect/verify | No writes or artifact access on manifest failure | Self-hash mismatch or precise parse/duplicate-key error; authenticity limitation documented | Category `manifest_integrity`/`manifest_parse`, field and safe action | 3 for corruption; 4 for unsupported semantics | Manifest changes cannot pass unnoticed by normal verification | Hashing only data files |
| RA-025 | `test_integrity.py::test_path_traversal_absolute_and_symlink_escape_are_rejected` | Containment checks implemented | Parameterize `../outside`, `/abs`, `C:\...`, UNC, encoded separators, in-root symlink/junction to sentinel | Inspect/load with open hook | Outside sentinel unchanged/unopened; no output | `UnsafeArtifactPathError`; normalized offending path and root reported | Contains role/path, `outside run directory`, remove/repair action | 3 | Artifact inspection is confined to run root | Directory traversal/junction escape |
| RA-026 | `test_integrity.py::test_interrupted_write_never_appears_succeeded` | Writer supports injectable failure at each commit phase | Minimal run, fail before/after result replace and before manifest replace | Re-open directory after each injected interruption | Only `running` manifest and/or ignored temp files; never terminal manifest referencing absent/bad file | Reader returns running/recovery state or precise incomplete error; no succeeded run | Inspect shows `running` and recovery guidance | 0 inspect or 3 for damaged running manifest, never success | False completed run after crash |
| RA-027 | `test_integrity.py::test_failed_run_loads_without_result_arrays` | Failure manifest writer/reader exists | Exception before arrays with redacted structured failure | Load/inspect; call `load_result` separately | `manifest.json`, environment when captured; no `result.npz` required | `load_run` succeeds with status/failure; `load_result` raises `NoUsableResult` with reason | Inspect shows stage/type/message/action; no traceback required | 0 inspect; 3 load-result CLI equivalent | Failure evidence is first-class, not “corrupt success” | Inability to inspect failed campaigns |

## Custom components — 6 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-028 | `test_custom_components.py::test_builtin_problem_and_operators_replay_by_stable_id` | Built-in descriptors/resolver exist | Rich built-in NSGA-II run | Exact replay | Complete new run | Resolver uses stable IDs and exact config; arrays exact | Lists resolved built-in IDs and exact result | 0 | Aliases/names do not alter built-in replay identity | Re-resolution through changed defaults |
| RA-029 | `test_custom_components.py::test_installed_plugin_replays_only_when_identity_matches` | Test entry-point registry and plugin descriptor protocol exist | Fake installed plugin with matching group/name/distribution/version/hash | Load then trusted normal plugin replay | New complete run | Loading invokes no plugin; replay resolves one matching entry point and succeeds at declared level | Shows verified plugin identity | 0 | Plugin code is delayed until replay and identity-checked | Eager entry-point execution or wrong plugin |
| RA-030 | `test_custom_components.py::test_missing_plugin_loads_result_but_replay_is_unavailable` | Plugin resolver reports absence | Same stored plugin run, plugin removed | `load_result`, inspect, reproduce | Original unchanged; no replay output | Arrays load; compatibility unavailable with exact missing distribution/entry point | Inspect succeeds and names install action; reproduce refuses | 0 inspect; 6 reproduce | Data access does not depend on plugin availability | All-or-nothing custom artifact loading |
| RA-031 | `test_custom_components.py::test_importable_custom_problem_protocol_requires_explicit_trust` | Versioned JSON custom protocol implemented | Importable test module/class, config, matching source hash | Load with import sentinel; replay without and with `trust_custom_code=True` | No output on refusal; new run on trust | Load executes no import; first replay refuses; trusted replay reconstructs config and respects level | First says `custom-code trust not granted`; second lists module/protocol/hash | 6 then 0 | Custom import is explicit and auditable | Code execution during load |
| RA-032 | `test_custom_components.py::test_lambda_or_closure_result_loads_but_is_manual` | Non-serializable descriptor path exists | Saved result from lambda/notebook-local closure with source hash/reason | Load, inspect, reproduce | Core stored result remains complete | Arrays load; declared level manual; reproduce requests user-supplied reconstruction and never evals stored text | Shows reason `no stable import path` and manual action | 0 inspect; 6 reproduce | Nonserializable code is recorded, not pickled/dropped | Silent metadata loss or unsafe cloudpickle |
| RA-033 | `test_custom_components.py::test_loading_custom_artifact_executes_no_imports_or_user_code` | Data-only reader isolated from resolver | Malicious module string, import hook/sentinel, constructor payload with code-like strings, unknown role | `load_run`, `load_result`, verify-only | Sentinels unchanged; no network/process/file side effect | Stored strings returned inertly; no import hook calls; unknown role unopened | Verification succeeds or reports inert unknown role; never prompts to execute | 0 | All normal load paths are data-only | Import/eval side effects from manifest strings |

## Public API and UX — 5 tests

| ID | Proposed module/function | Precondition | Fixture | Operation | Expected files | Expected Python result | Expected CLI output | Exit | Invariant | Defect prevented |
|---|---|---|---|---|---|---|---|---:|---|---|
| RA-034 | `test_public_api_cli.py::test_public_save_load_preserves_result_and_manifest_access` | Top-level facade exports implemented | Result containing public `F/X`, `data` arrays/counters, meta provenance | `vamos.save_result`; `vamos.load_result`; `vamos.load_run` | Complete v1 core files | All contract-covered public fields equal; immutable manifest accessible; return types match annotations | N/A | N/A | Basic and advanced Python paths share one reader | UX helper diverging from canonical storage |
| RA-035 | `test_public_api_cli.py::test_loading_and_replay_are_distinct_operations` | Public functions instrumentable | Complete exact run, optimization call spy | Call load APIs, then reproduce | Load creates/writes nothing; reproduce creates new attempt only | Load spy count zero; reproduce count one; source unchanged | Help and inspect text say `loading stored data does not execute replay` | 0 | API names correspond to trust/side-effect boundaries | Accidental optimization during inspection |
| RA-036 | `test_public_api_cli.py::test_verify_only_performs_no_optimization` | CLI commands implemented | Complete run and subprocess-visible optimization sentinel | `vamos reproduce RUN --verify-only --json` | No new run/output files | N/A; sentinel proves no algorithm/plugin resolution | JSON integrity/effective level, `executed=false` | 0 | Verification is read-only and code-free | “Verify” that reruns the experiment |
| RA-037 | `test_public_api_cli.py::test_errors_are_actionable_in_human_and_json_modes` | Structured error renderer exists | Parameterize corrupt hash, unknown major, env mismatch, missing plugin, output collision | Invoke corresponding Python and CLI operations | No destructive writes | Typed error fields populated consistently | Each mode includes operation, role/field, reason, expected, action; exit matches 3/4/5/6/8 | 3/4/5/6/8 | Users can safely recover without reading source | Cryptic traceback-only failures |
| RA-038 | `test_public_api_cli.py::test_existing_save_result_import_has_compatibility_path` | UX alias and top-level export wired | Existing call style `from vamos.ux.api import save_result` and caller ignoring return | Save then load via top-level API; inspect deprecation policy | Complete v1 files and optional CSV views | Old import/call succeeds; returned `StoredRun` may be ignored; docs/signature point to canonical behavior | If warning exists, it names stable replacement and removal window; no warning required for permanent alias | 0 when CLI inspection used | Existing public callers are not abruptly broken | Unannounced saver replacement |

## Security/resource-limit additions to the rows

RA-024, RA-025, and RA-033 MUST be parameterized to include corrupted JSON,
duplicate keys, malicious component strings, unknown roles, absolute/traversal
paths, and symlink/junction escape. RA-004 or a companion parameterization in
RA-023 MUST include malformed NPY headers, object dtype, excessive dimensions,
too many ZIP members, declared array bytes over the configured limit, excessive
total uncompressed bytes, and suspicious compression ratio. Each case must fail
before unsafe allocation or code execution.

These parameterizations do not increase the stable test-ID count; reports may
show individual pytest cases beneath the same acceptance ID.

## Coverage by contract decision

| Contract area | Acceptance IDs |
|---|---|
| Requested vs resolved ownership | RA-002, RA-003, RA-007, RA-016 |
| Manifest/schema/versioning | RA-008, RA-019, RA-020, RA-024 |
| Numerical bundle | RA-001, RA-004–RA-006, RA-022, RA-023 |
| Relocation/path safety | RA-021, RA-025 |
| Atomic status/failure | RA-026, RA-027 |
| Replay levels/environment | RA-009–RA-014 |
| Legacy policy | RA-015–RA-020, RA-038 |
| Custom components/trust | RA-028–RA-033 |
| Public Python/CLI semantics | RA-034–RA-038 |

## Completion rule

An implementation pull request may claim only the acceptance IDs it executes.
A later passing integration test cannot compensate for a missing invariant if
its fixture does not exercise that invariant. The full run-artifact feature is
complete only when RA-001 through RA-038 pass on supported Windows and Linux
CI, with path/symlink cases adapted to platform capability and never silently
skipped when the platform supports the feature.

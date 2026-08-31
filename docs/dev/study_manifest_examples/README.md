# StudyManifest v1 machine-readable examples

These ten sanitized JSON fixtures make representative contract states and
failures reviewable before production implementation. They are validated by
`tests/docs/test_study_manifest_contract.py`.

Each file is a documentation envelope with identity
`vamos.study-contract-example` version `1.0.0`. The envelope is not a supported
persistence document. Its `canonical_files` array models a relocatable virtual
study tree: every entry contains a root-relative POSIX path, its JSON document,
and the exact byte length and SHA-256 of the sorted, compact, UTF-8 canonical
form. Every document also carries its semantic self-hash, calculated before the
`integrity` member is added.

Compact `vamos.run-manifest` entries intentionally omit RunManifest-owned
specifications and artifact bodies. Their `fixture_compaction` marker delegates
those details to the independently tested [run-artifact
contract](../run_artifact_contract.md); the study examples exercise only the
frozen reference identity, path, byte, file-hash, and semantic-hash boundary.
Operational lock/lease paths, when relevant, appear separately and are not
canonical scientific files.

| Fixture | Contract state or failure |
|---|---|
| `01-empty-created.json` | Valid atomically created empty study. |
| `02-running.json` | Valid running task and attempt with an operational lease. |
| `03-succeeded.json` | Valid selected run and completed study. |
| `04-completed-with-failures.json` | Valid continue-policy terminal partial study. |
| `05-fail-fast-paused.json` | Valid failed task, pending task, and paused study. |
| `06-interrupted-attempt.json` | Valid reconciled interrupted attempt. |
| `07-retried-task.json` | Valid immutable failed attempt followed by success. |
| `08-relocated.json` | Valid completed tree whose IDs survive relocation. |
| `09-invalid-transition.json` | Valid completed tree plus a requested forbidden transition; expects exactly `INVALID_STATE_TRANSITION`. |
| `10-corrupt-run-reference.json` | One deliberately wrong referenced RunManifest file hash; expects exactly `RUN_MANIFEST_HASH_MISMATCH`. |

The invalid-transition fixture is structurally sound before its declared
operation. The corrupt-reference fixture recomputes every surrounding document
and inventory hash after inserting the one bad reference, so it cannot pass or
fail for an incidental second reason.

# Save and load Python run artifacts

VAMOS can persist one Python optimization result as a relocatable v1 run
directory. The manifest and numerical bundle are authoritative; CSV files are
compatibility views for existing analysis workflows.

```python
import vamos

result = vamos.optimize(
    "zdt1",
    algorithm="nsgaii",
    pop_size=40,
    max_evaluations=400,
    seed=7,
)

stored = vamos.save_result(result, "runs/zdt1-seed-7")
loaded = vamos.load_result("runs/zdt1-seed-7")
run = vamos.load_run("runs/zdt1-seed-7")

print(stored.manifest.run_id)
print(loaded.F.shape)
print(run.status, run.environment["python"])
```

`save_result` returns an immutable `StoredRun`. Existing callers that import
`save_result` from `vamos.ux.api` and ignore its return value remain valid; that
name is a permanent alias of the canonical saver.

## Directory contents

A successful save publishes the directory only after every final artifact is
complete:

```text
runs/zdt1-seed-7/
├── manifest.json       # schema, specs, provenance, outcome, hashes
├── result.npz          # authoritative numerical arrays
├── environment.json    # bounded, privacy-conscious environment snapshot
├── FUN.csv             # non-authoritative compatibility view
├── X.csv               # non-authoritative compatibility view
└── metadata.json       # non-authoritative compatibility view
```

`result.npz` preserves numerical values, shapes, dtype widths, and byte order.
It can contain `F`, `X`, `G`, `CV`, final population/archive arrays, and material
reference directions. V1 permits only fixed-width boolean, signed integer,
unsigned integer, and floating dtypes. Object, string, structured, complex,
datetime, and pickle-backed arrays are rejected.

Every referenced file has an exact byte length and SHA-256 digest. The manifest
also has a canonical semantic self-hash. All stored paths are normalized
relative POSIX paths, so the complete directory can be moved and loaded from a
new location.

## Loading and verification

Loading is a data-only operation. It performs no optimization, component/plugin
resolution, custom imports, shell commands, or network access. Replay and
reproduction are deliberately outside this v1 core and are not available from
these functions.

Both load functions accept a verification mode:

- `verify="manifest"` validates the JSON schema and manifest self-hash. An
  accessed numerical/environment artifact is still parsed using safe bounded
  readers.
- `verify="required"` (default) also verifies byte length and SHA-256 for
  artifacts required by the requested load operation.
- `verify="all"` verifies every known referenced artifact, including CSV
  compatibility views. Unknown extension roles remain inert.

`load_run` exposes the immutable manifest immediately and loads `.result` and
`.environment` lazily. `load_result` is the convenience path and attaches the
same manifest as `result.manifest`. A failed run can be inspected with
`load_run`; asking it for `.result` raises an actionable `IncompleteRunError`.

Legacy count-only CSV directories are recognized and actionably rejected by
this reader. Legacy loading and migration are separate, deferred work; normal
loading never fabricates missing provenance or rewrites an old directory.

Current limitations are intentional: there is no legacy loader/migrator, no
CLI or Python reproduction command, and no custom-code/plugin replay. Stored
custom-component descriptions remain inert data during loading.

## Defensive limits

Normal readers use finite defaults:

| Limit | Default |
|---|---:|
| Manifest JSON | 8 MiB |
| Environment JSON | 16 MiB |
| One artifact / one array | 512 MiB |
| Artifact descriptors / ZIP members / arrays | 128 / 128 / 64 |
| Total uncompressed array bytes | 1 GiB |
| Total array elements | 100 million |
| NPY header / JSON depth | 64 KiB / 64 |
| Compression ratio | 1000:1 |

A trusted caller may explicitly supply a different `vamos.LoadLimits` instance:

```python
limits = vamos.LoadLimits(max_artifact_bytes=768 * 1024 * 1024)
loaded = vamos.load_result("runs/large", limits=limits)
```

Limits are never increased automatically after a rejection. NPZ files are
inspected for member count, overlap, header consistency, dtype, declared size,
and compression ratio before NumPy allocates their arrays, and are always
materialized with `allow_pickle=False`.

## Non-destructive writes

The destination path must not exist. Empty, valid, unrelated, and partially
written directories are all collisions and are never overwritten or merged.
The writer snapshots supported arrays before I/O, writes into a uniquely owned
sibling staging directory, fsyncs artifacts, commits the terminal manifest
last, and renames the complete directory into place. A write failure removes
only its owned staging directory and never exposes a succeeded destination.

SHA-256 detects accidental corruption; it is not a digital signature. Restore
missing or modified canonical files from the original run rather than falling
back to CSV or editing hashes.

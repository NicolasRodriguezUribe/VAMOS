# Save and load run artifacts

VAMOS writes one relocatable canonical v1 directory for each run.

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
loaded = vamos.load_result(stored.root)
run = vamos.load_run(stored.root)

print(stored.manifest.run_id)
print(loaded.F.shape)
print(run.status, run.environment["python"])
```

## Directory contents

```text
runs/zdt1-seed-7/
├── manifest.json
├── result.npz
└── environment.json
```

The manifest contains requested intent, resolved execution state, the actual
seed, provenance, outcome, artifact hashes, and a semantic self-hash. The NPZ
contains numerical result/population/archive arrays. The environment document
contains a bounded privacy-conscious runtime snapshot.

The directory may be moved as a unit. All references are relative and confined
to it. The destination passed to `save_result` must not exist; VAMOS never
overwrites or merges a run.

## Requested and generated seeds

An explicit seed, including zero, is used unchanged:

```python
result = vamos.optimize("zdt1", seed=0)
print(result.meta["seed"])  # 0
```

Pass `None` to ask VAMOS to generate a seed. Generation happens before any
stochastic execution object is constructed:

```python
result = vamos.optimize("zdt1", seed=None)
actual_seed = result.meta["seed"]
stored = vamos.save_result(result, "runs/generated-seed")

print(stored.manifest.requested_spec["defaults"]["seed"])  # None
print(stored.manifest.resolved_spec["seed"])                # actual integer
```

## Manual results

`OptimizationResult` instances created by `vamos.optimize` already carry their
complete run context. A manually constructed result must provide both the
requested and resolved specs:

```python
stored = vamos.save_result(
    manual_result,
    "runs/manual",
    requested_spec=requested_spec,
    resolved_spec=resolved_spec,
)
```

Without both mappings, saving raises `vamos.IncompleteRunMetadataError`. VAMOS
does not invent a seed or execution configuration from arrays.

## Loading and verification

Loading is data-only. It does not rerun optimization, resolve plugins, import
recorded custom components, use pickle, invoke a shell, or access the network.
Replay/reproduction is future work and is intentionally separate from loading.

Verification modes are:

- `verify="manifest"`: validate manifest syntax, semantics, task ID, and
  self-hash;
- `verify="required"` (default): also verify artifacts required for loading;
- `verify="all"`: verify every known referenced artifact.

`load_run` returns an immutable manifest and lazy `.result`/`.environment`
access. `load_result` is the convenient numerical path. A failed run remains
inspectable through `load_run`, while `.result` raises `IncompleteRunError`.

## Numerical safety and limits

NPZ loading always uses `allow_pickle=False`. V1 accepts fixed-width boolean,
integer, unsigned integer, and floating arrays. It rejects executable/object,
string, structured, complex, and datetime dtypes.

Readers validate bounded JSON, artifact count/size, ZIP member layout, NPY
headers, shapes, dtype, total elements/uncompressed bytes, and compression
ratio before materializing arrays. A trusted caller can explicitly provide a
different `vamos.LoadLimits`; limits are never raised automatically.

## Unsupported directories

VAMOS is pre-release and supports only `vamos.run-manifest` `1.0.0`. A directory
without that manifest is rejected with guidance to regenerate the run using the
current version. There are no fallback readers, format detectors, or migration
aliases.

SHA-256 detects accidental modification; it is not a digital signature.

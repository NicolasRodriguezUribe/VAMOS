# Experiment blocks: stopping + bounded archive

This project supports method-level early stopping and bounded archiving. These are not feature toggles:
they define explicit contracts (artifacts + metadata) and are evaluated experimentally.

## stopping.hv_convergence

Enable HV-based convergence stopping driven by a hypervolume trace sampled during the run.

Example:

```yaml
stopping:
  hv_convergence:
    enabled: true
    every_k: 200
    window: 10
    patience: 5
    epsilon: 1e-4
    epsilon_mode: rel     # abs|rel
    statistic: median     # mean|median|min
    min_points: 25
    confidence: null      # e.g. 0.95 to enable bootstrap CI
    bootstrap_samples: 300
    ref_point: [2.0, 2.0] # must match n_obj (or use "auto")
```

Artifacts:
- `hv_trace.csv` (see `experiments/ARTIFACT_CONTRACT.md`)

Metadata:
- `metadata.json` additions under `stopping`

Notes:
- For 2 objectives, HV is computed exactly.
- For >2 objectives, HV may be unavailable unless a backend provides it; trace rows log reason codes.
- Use `ref_point: "auto"` to let the runner derive a reference point from current data.

## archive.bounded

Enable bounded archive maintenance with explicit pruning policies.

Example:

```yaml
archive:
  bounded:
    enabled: true
    archive_type: size_cap     # size_cap|epsilon_grid|hvc_prune|hybrid
    size_cap: 200
    nondominated_only: true
    prune_policy: crowding     # crowding|hv_contrib|random|mc_hv_contrib
    epsilon: 0.01              # grid resolution for epsilon_grid/hybrid
    hv_ref_point: null         # optional; required for hv_contrib policies
    hv_samples: 20000          # for mc_hv_contrib
    rng_seed: 0
```

Artifacts:
- `archive_stats.csv` (see `experiments/ARTIFACT_CONTRACT.md`)

Metadata:
- `metadata.json` additions under `archive`

## Reproducibility

Runs should be launched with fixed seeds and fixed budgets. Early stopping changes executed evaluations,
but the run still reports the original max budget in config. Use `stopping.evals_stop` and `hv_trace.csv`
for analysis.

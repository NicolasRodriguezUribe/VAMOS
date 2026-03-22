# Experiment blocks: stopping + external archive

This project supports method-level early stopping and external archive tracking. These are not feature toggles:
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

## archive.external

Enable external archive maintenance with explicit pruning policies.

Example:

```yaml
archive:
  external:
    enabled: true
    capacity: 200
    truncate_size: 200
    pruning: crowding          # crowding|hv|mc_hv|knn|maxmin|ref_dirs
    hv_ref_point: null         # optional; required for hv-based policies
    rng_seed: 0
    objective_tolerance: 1.0e-10
    deduplicate_in: objective  # objective|decision|both
    decision_tolerance: 1.0e-32
```

Artifacts:
- `archive_stats.csv` (see `experiments/ARTIFACT_CONTRACT.md`)

Metadata:
- `metadata.json` additions under `archive`

Notes:
- In the tuning spaces, external archives use the population size as their default capacity.
- When an algorithm is configured with an external archive, top-level results come from that archive by default unless `result_mode="population"` is requested.
- `pruning: hv` uses exact HV contributions in 2D and, when `moocore` is installed, exact higher-dimensional contributions as well. `mc_hv` always uses the Monte Carlo proxy.

## Reproducibility

Runs should be launched with fixed seeds and fixed budgets. Early stopping changes executed evaluations,
but the run still reports the original max budget in config. Use `stopping.evals_stop` and `hv_trace.csv`
for analysis.

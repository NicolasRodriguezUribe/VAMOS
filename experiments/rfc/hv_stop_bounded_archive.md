# RFC: HV-based stopping + bounded archive (TEVC-grade)

## Scope
Promote early stopping and archiving from a 'flag' to a *method* with:
- formal definitions (monitor + archive)
- explicit contracts (trace + metadata)
- algorithm-agnostic hook interface
- rigorous evaluation (time/quality trade-offs, sensitivity, robustness)

## A) HVConvergenceMonitor (method)
We observe a hypervolume trace sampled every `every_k` evaluations:
  HV_t = HV(A_t; r)
where A_t is the reference set used to compute HV at time t (typically current nondominated set or archive)
and r is a fixed reference point.

### Parameters (maximal, but configurable)
- every_k: sampling frequency
- window: rolling window length
- patience: number of consecutive windows with insufficient improvement before stopping
- epsilon: minimum meaningful improvement threshold
- epsilon_mode: abs | rel  (rel uses epsilon * |HV| scale)
- statistic: mean | median | min   (how to summarize window improvements)
- min_points: minimum trace points before decisions are allowed
- confidence (optional): CI level for a bootstrap test on window improvements
- bootstrap_samples: number of bootstrap resamples

### Decision rule (one representative robust variant)
Let Δ_t = HV_t - HV_{t-window}. Stop if, for `patience` consecutive checks:
  stat(Δ_t) <= epsilon_abs
Optionally require a bootstrap upper CI bound <= epsilon_abs.

Outputs:
- hv_trace.csv with evals/hv/hv_delta/stop_flag/reason
- metadata.json stopping.* keys

## B) BoundedArchive (method)
Maintain an archive with explicit boundedness and pruning contracts.

### Parameters
- enabled
- archive_type: size_cap | epsilon_grid | hvc_prune | hybrid
- size_cap: maximum archive size
- epsilon: grid resolution (for epsilon_grid)
- prune: crowding | hv_contrib | random | montecarlo_hv_contrib
- hv_ref_point: reference for contribution pruning
- hv_samples: samples for montecarlo contributions (if used)
- nondominated_only: keep only ND in archive (recommended)

Outputs:
- archive_stats.csv (insert/prune events)
- metadata.json archive.* keys

## C) Hook interface (algorithm-agnostic)
Expected callbacks:
- on_generation_end(evals, population, nd_set, archive)
- on_archive_update(evals, archive_stats)
- should_stop(evals) -> bool

## D) Evaluation plan (maximal)
1) Baselines:
  - no-archive/no-stop
  - bounded-archive only
  - bounded-archive + hv-stop
2) Metrics:
  - HV / IGD+ final (same max-evals budget)
  - runtime, evals-executed ratio, overhead of HV computation
3) Robustness:
  - sensitivity sweep: window, patience, epsilon, every_k, size_cap, grid epsilon
  - ref point robustness: 2-3 ref strategies
4) Statistics:
  - Wilcoxon + Holm and/or Friedman + Nemenyi over HV/IGD+
  - 'cost ablation': time saved vs quality delta

This RFC is implemented as:
- src/vamos/monitoring/hv_convergence.py
- src/vamos/archive/bounded_archive.py
and integrated via hooks discovered by experiments/scripts/inspect_hv_archive_hooks.py.

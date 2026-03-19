# Algorithms and backends

Algorithms (internal)
---------------------

- NSGA-II: continuous, permutation, binary, integer, mixed; supports archive, adaptive operators, HV early-stop.
  - `result_mode` accepts only `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
  - When archive is enabled, results still include `result["archive"]` alongside `result["population"]`.
  - `archive_mode="off"` keeps baseline NSGA-II behavior.
  - `archive_mode="passive"` keeps a separate exact unbounded archive without changing mating, survival, or the primary top-level result. The archive payload exposes the full archive plus a crowding-selected subset of size `archive_subset_size` (default: `pop_size`).
  - `archive_mode="hybrid_survival"` keeps NSGA-II ranking/full-front acceptance unchanged and changes only split-front truncation using `archive_hybrid_alpha` (default `0.5`) and objective-space archive `k`-NN novelty with `archive_hybrid_k` (default `3`). The current normalization mode is `archive_hybrid_normalization="minmax_archive_split"`.
  - `hybrid_survival` is currently supported only on the standard generational unconstrained path; constrained and steady-state/incremental runs fall back to the standard survival path.
  - Archive-family runs expose compact execution metadata in `result["archive_diagnostics"]`, including whether passive or hybrid mode ran, whether hybrid fell back, and whether split-front novelty used archive or local-only scoring.
  - Archive-assisted mating is not implemented.
  - Supported external-archive prune policies are `crowding`, `hv`, `mc_hv`, `knn`, `maxmin`, and `ref_dirs`.
  - `hv` uses exact hypervolume contributions in 2D and exact higher-dimensional contributions when `moocore` is available; `mc_hv` keeps the Monte Carlo approximation path.
- NSGA-III: many-objective real/binary/integer; reference direction support. Matching `pop_size` to the number of reference directions is recommended (with divisions p: `comb(p + n_obj - 1, n_obj - 1)`); mismatches emit a warning unless strict enforcement is enabled.
- MOEA/D: real/binary/integer; aggregation methods (tchebycheff, weighted sum, pbi). Defaults align with jMetalPy (PBI aggregation, DE crossover CR=1.0/F=0.5, packaged weight vectors for n_obj > 2).
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- SMS-EMOA: real/binary/integer; adaptive reference points.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- SPEA2: real/binary/integer with constraint handling.
- IBEA: epsilon or hypervolume indicator variants.
- SMPSO: real-coded, archive support.
- AGE-MOEA: adaptive geometry estimation for many-objective search.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.
- RVEA: reference-vector guided many-objective search.
  - `result_mode` accepts `non_dominated` (default) or `population`.
  - External archive configuration (`.external_archive(...)`) becomes the default result source unless you explicitly set `result_mode="population"`.

Enable AOS
----------

Adaptive Operator Selection (AOS) is an opt-in layer for NSGA-II. See
`docs/paper/aos-method.md` for the contract and
`examples/configs/nsgaii_aos_min.yml` for a runnable example.

Optional baselines (install extras)
-----------------------------------

- PyMOO NSGA-II (real and permutation), jMetalPy NSGA-II (real and permutation), PyGMO NSGA-II.
- Enabled via `--include-external` and extras `research`.

Backends
--------

- NumPy (default): vectorized CPU kernels.
- Numba: JIT acceleration for supported kernels (set `VAMOS_USE_NUMBA_VARIATION=1` for permutation/binary/integer variation).
- MooCore: accelerated kernels via `moocore` (install `compute` extra).

Live visualization
------------------

Enable `--live-viz` to stream Pareto fronts during runs (`--live-viz-interval`, `--live-viz-max-points`). Saves a `live_pareto.png` at run end.

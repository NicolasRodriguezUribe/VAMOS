# Algorithms and backends

Algorithms (internal)
---------------------

- NSGA-II: continuous, permutation, binary, integer, mixed; supports archive, adaptive operators, HV early-stop.
- NSGA-III: many-objective real/binary/integer; reference direction support. Matching `pop_size` to the number of reference directions is recommended (with divisions p: `comb(p + n_obj - 1, n_obj - 1)`); mismatches emit a warning unless strict enforcement is enabled.
- MOEA/D: real/binary/integer; aggregation methods (tchebycheff, weighted sum, pbi). Defaults align with jMetalPy (PBI aggregation, DE crossover CR=1.0/F=0.5, packaged weight vectors for n_obj > 2).
- SMS-EMOA: real/binary/integer; adaptive reference points.
- SPEA2: real/binary/integer with constraint handling.
- IBEA: epsilon or hypervolume indicator variants.
- SMPSO: real-coded, archive support.

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

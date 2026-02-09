# Problems registry

General
-------

Use `--problem <key>` or `--problem-set <preset>`. Override dimensions with `--n-var`/`--n-obj` if the spec allows.

Continuous benchmarks
---------------------

- ZDT: zdt1, zdt2, zdt3, zdt4, zdt6 (bi-objective, continuous).
- DTLZ: dtlz1-dtlz7 (default 3 objectives, override allowed).
- WFG: wfg1-9 (requires `research` extra, override objectives allowed).
- LZ09: lz09_f1-f9.
- CEC2009: cec2009_uf1-cec2009_uf10, cec2009_cf1.
- LSMOP: lsmop1-lsmop9 (large-scale, default 300 variables).
- Constrained many-objective: c1dtlz1, c1dtlz3, c2dtlz2, c3dtlz1, dc1dtlz1, dc1dtlz3, dc2dtlz1, dc2dtlz3, dc3dtlz1, dc3dtlz3, mw1-mw14.

DTLZ benchmarks note
--------------------

Standard DTLZ settings use:
- DTLZ1: `n_var = n_obj + 4` (`k=5`)
- DTLZ2-6: `n_var = n_obj + 9` (`k=10`)
- DTLZ7: `n_var = n_obj + 19` (`k=20`)

Non-standard `n_var` values are allowed, but VAMOS will warn so results are not accidentally compared against the canonical setting.

Permutation benchmarks
----------------------

- tsp6 (toy)
- kroa100, krob100, kroc100, krod100, kroe100 (TSPLIB)

Binary benchmarks
-----------------

- zdt5 (bi-objective, binary).
- bin_feat (feature selection surrogate)
- bin_knapsack
- bin_qubo

Integer benchmarks
------------------

- int_alloc (resource allocation)
- int_jobs (job assignment)

Mixed and real-world
--------------------

- mixed_design (mixed real/integer/categorical)
- ml_tuning (SVM hyperparameter tuning; needs scikit-learn via `examples` or `analysis` extras)
- welded_beam (mixed constrained design)
- fs_real (binary feature selection on real data; needs scikit-learn)

Presets
-------

- `families` preset in CLI covers representative ZDT/DTLZ/WFG cases.
- `cec` includes CEC2009 UF1-10 + CF1.
- `lsmop` includes LSMOP1-9.
- `constrained_many` includes C-DTLZ, DC-DTLZ, and MW constrained families.

Reference fronts
----------------

Built-in CSVs exist for ZDT problems (including `zdt5`). For custom problems set `--hv-reference-front` (CSV with two columns) when using `--hv-threshold`.

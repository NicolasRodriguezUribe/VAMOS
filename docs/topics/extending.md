# Extending VAMOS

Algorithms
----------

- Add new classes under `src/vamos/engine/algorithm/`, register in `algorithm/registry.py`, and expose defaults in `algorithm/config.py`/`ExperimentConfig` if needed.
- Use existing kernels where possible; follow existing patterns for `run()` returning `{"X": ..., "F": ..., "archive": ...}`.
- Add smoke tests under `tests/` (see `test_algorithms_smoke.py`).

Operators and kernels
---------------------

- Operators live in `src/vamos/operators/impl/` (real, permutation, binary, integer, mixed). Add new operators with RNG-friendly vectorized implementations and register in `operators/impl/registry.py`. Algorithm-specific wiring lives in `src/vamos/operators/policies/`.
- Kernels live in `src/vamos/foundation/kernel/`; register new backends in `kernel/registry.py` and mirror the NumPy API.

Problems
--------

- Add problem classes under `src/vamos/foundation/problem/`.
- Register specs in `src/vamos/foundation/problem/registry/families/<family>.py` (not in `specs.py`).
  See `src/vamos/foundation/problem/registry/AGENTS.md` for the canonical workflow.
- Ensure `evaluate(X, out)` fills `out["F"]` (and `out["G"]` if constrained) in vectorized form.

Config and CLI
--------------

- For new CLI flags or config keys, update `src/vamos/cli.py` and `src/vamos/experiment_config.py`.
- Keep YAML/JSON specs aligned with CLI defaults; add examples when new knobs appear.

Documentation and tests
-----------------------

- Update relevant docs pages when adding public features.
- Add pytest coverage mirroring `src` layout; include determinism checks for stochastic pieces where practical.
- Run `ruff check src tests`, `black src tests`, and `pytest` before opening a PR.

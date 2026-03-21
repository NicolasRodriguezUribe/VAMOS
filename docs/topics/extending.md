# Extending VAMOS

This is the canonical extension guide. Keep other contributor docs thin and point back here.

Algorithms
----------

- Add new classes under `src/vamos/engine/algorithm/`.
- Register them in `src/vamos/engine/algorithm/registry.py`.
- Put typed config builders under `src/vamos/engine/algorithm/config/` when you need a reproducible public config object.
- Use existing kernels where possible; follow existing patterns for `run()` returning `{"X": ..., "F": ..., "archive": ...}`.
- Add smoke tests under `tests/` (see `test_algorithms_smoke.py`).
- Use `docs/dev/add_algorithm.md` for a concrete step-by-step template.

Operators and kernels
---------------------

- Operators live in `src/vamos/engine/operators/impl/` (real, permutation, binary, integer, mixed). Add new operators with RNG-friendly vectorized implementations and register in `operators/impl/registry.py`. Algorithm-specific wiring lives in `src/vamos/engine/operators/policies/`.
- Kernels live in `src/vamos/foundation/kernel/`; register new backends in `kernel/registry.py` and mirror the NumPy API.

Problems
--------

- Add problem classes under `src/vamos/foundation/problem/`.
- Register specs in `src/vamos/foundation/problem/registry/families/<family>.py` (not in `specs.py`).
  See `src/vamos/foundation/problem/registry/AGENTS.md` for the canonical workflow.
- Ensure `evaluate(X, out)` fills `out["F"]` (and `out["G"]` if constrained) in vectorized form.

Config and CLI
--------------

- For new CLI flags or config keys, update `src/vamos/experiment/cli/` and `src/vamos/foundation/core/experiment_config.py`.
- Keep YAML/JSON specs aligned with CLI defaults; add examples when new knobs appear.
- Prefer the high-level `optimize(...)` API in public examples. Use `algorithm_config` objects only when the extra control is essential.

Documentation and tests
-----------------------

- Update relevant docs pages when adding public features.
- Add pytest coverage mirroring `src` layout; include determinism checks for stochastic pieces where practical.
- Run `ruff check src tests`, `ruff format src tests`, and `pytest` before opening a PR.


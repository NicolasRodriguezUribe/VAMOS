# Architecture Health

Purpose: prevent future refactors by enforcing layer boundaries and limiting module bloat.
These rules are guardrails for long-term maintainability in a research-oriented codebase.

## Health Gates (run locally)
- `pytest -q tests/architecture/test_layer_boundaries.py`
- `pytest -q tests/test_monolith_guard.py`
- `pytest -q tests/test_public_api_guard.py`
- `pytest -q tests/test_import_time_smoke.py`
- `pytest -q tests/test_no_deprecation_shims.py`
- `pytest -q tests/test_no_prints_in_library.py`
- `pytest -q tests/test_optional_deps_policy.py`
- `pytest -q tests/test_logging_policy.py`
- `pytest -q`

## Layering Policy (current reality)
- foundation must not import engine/ux/experiment.
- engine may depend on foundation and hooks only.
- experiment and ux may depend on foundation and engine.
- Facades: prefer `vamos.api`, `vamos.engine.api`, `vamos.ux.api`, `vamos.experiment.quick`.

## Optional Dependencies Policy
- foundation/** and engine/**: no top-level imports of optional/heavy deps.
- experiment/external/**: integration boundary for optional deps; imports must be lazy or guarded.
- ux/studio/**: streamlit allowed only here.
- Dynamic import loopholes (`importlib.import_module`, `__import__`) are disallowed at top-level.

## No Monoliths Policy
- File size thresholds: core <= 450 LOC, CLI/UI <= 350 LOC.
- Function size <= 250 LOC; class size <= 400 LOC.
- Allowlists are forbidden. Split instead.
- Split pattern: create a package with focused modules and keep orchestration thin.

## Logging/Printing Policy
- No `print()` in library code (allowed only in CLI/UI).
- No `logging.basicConfig()` in library modules.
- CLI logging config happens at invocation only via local handlers.

## Adding Problems/Operators (no cross references)
- Problem registry: add specs in `foundation/problem/registry/families/*.py`.
- Operators: canonical package is `vamos.operators`.
- Update docs/tests when adding new modules or APIs.

# Architecture Health

Purpose: prevent future refactors by enforcing layer boundaries and limiting module bloat.
These rules are guardrails for long-term maintainability in a research-oriented codebase.

## Canonical Decisions (ADRs)
- Read before any architectural change: `docs/dev/adr/index.md`.
- Mandatory ADRs: layering/facades, import-time purity, optional deps, no shims, health gates/retention.

## Health Gates (run locally)
- `python tools/health.py` (uses the mypy error budget gate)
- `python tools/health.py --mypy-full` (runs raw mypy; stricter than CI)
- `pytest -q tests/architecture/test_layer_boundaries.py`
- `pytest -q tests/test_monolith_guard.py`
- `pytest -q tests/test_public_api_guard.py`
- `pytest -q tests/test_import_time_smoke.py`
- `pytest -q tests/architecture/test_no_import_time_side_effects.py`
- `pytest -q tests/architecture/test_public_api_snapshot.py`
- `pytest -q tests/architecture/test_dependency_policy.py`
- `pytest -q tests/architecture/test_report_retention_policy.py`
- `pytest -q tests/test_no_deprecation_shims.py`
- `pytest -q tests/test_no_prints_in_library.py`
- `pytest -q tests/test_optional_deps_policy.py`
- `pytest -q tests/test_logging_policy.py`
- `pytest -q`

## Layering Policy (current reality)
- foundation must not import engine/ux/experiment.
- engine may depend on foundation and hooks only.
- experiment and ux may depend on foundation and engine.
- Facades: prefer `vamos.api`, `vamos.engine.api`, `vamos.ux.api`.

## Optional Dependencies Policy
- foundation/** and engine/**: no top-level imports of optional/heavy deps.
- experiment/external/**: integration boundary for optional deps; imports must be lazy or guarded.
- ux/studio/**: streamlit allowed only here.
- Dynamic import loopholes (`importlib.import_module`, `__import__`) are disallowed at top-level.
- Dependency list is enforced by `tests/architecture/test_dependency_policy.py`.

## Public API Snapshot
- Public facades are frozen via `tests/architecture/test_public_api_snapshot.py`.
- Update the snapshot intentionally with `python tools/update_public_api_snapshot.py`.

## Import-Time Purity
- No executable calls at module import time (top-level `ast.Call` outside `TYPE_CHECKING`/`__main__` blocks).
- Move initialization into functions or CLI entrypoints; use lazy factories for registries.
- Avoid top-level env reads or dynamic import calls; perform them inside runtime functions.

## Report Retention
- Keep at most 5 `reports/final_audit_*.md` files in `reports/`; move older ones to `reports/archive/`.
- Keep at most 5 `reports/final_audit_*_artifacts/` directories in `reports/`.
- `final_audit_latest.md` at repo root must match the newest report under `reports/`.
- No other `final_audit_*.md` files are allowed at repo root.
- Keep `reports/` markdown size under 15 MB (excluding `reports/archive/`).
- Keep `reports/archive/` capped at 20 files; prune older audits when needed.
- Raw outputs (mypy/ruff/build logs) must live under `reports/<audit>_artifacts/`.

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
- Operators: implementations live in `vamos.operators.impl`, algorithm wiring in `vamos.operators.policies`.
- Update docs/tests when adding new modules or APIs.

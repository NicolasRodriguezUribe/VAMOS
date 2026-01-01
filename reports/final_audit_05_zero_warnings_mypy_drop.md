# Final Audit 05 - Zero Warnings + Mypy Drop

## Summary
- Eliminated the matplotlib `FigureCanvasAgg` warning by making live visualization ticks backend-aware (no `plt.pause` on non-interactive backends).
- Reduced mypy errors from **843** to **761** while focusing fixes on the requested targets (`jmetalpy.py`, `moocore_backend.py`, `variation/pipeline.py`) plus small typing cleanups.
- Updated the mypy error budget to `max_errors: 761` and kept all architecture health gates green.
- Removed the Black format check from CI (Ruff format is the authoritative formatter).
- Full test suite runs with **0 warnings**.

## Warning removal (live visualization)
- Changed `LiveParetoPlot` and `LiveTuningPlot` to detect non-interactive backends and use `draw_idle()`/`flush_events()` instead of `plt.pause()`.
- This preserves interactive behavior for interactive backends while preventing the headless Agg warning.

## Mypy progress
- Before: `Found 843 errors in 164 files` (from `reports/final_audit_04_perfect_finish.md`).
- After:  `Found 761 errors in 157 files` (from `reports/mypy_latest.txt`).
- Error budget updated: `tests/architecture/mypy_error_budget.json` -> `max_errors: 761`.

Top offenders (before):
- `src/vamos/experiment/external/jmetalpy.py`: 27
- `src/vamos/foundation/kernel/moocore_backend.py`: 25
- `src/vamos/engine/algorithm/components/variation/pipeline.py`: 21
- `src/vamos/engine/algorithm/builders.py`: 21
- `src/vamos/ux/studio/app.py`: 21
- `src/vamos/engine/algorithm/moead/operators.py`: 20
- `src/vamos/operators/permutation.py`: 18
- `src/vamos/experiment/external/pymoo.py`: 17
- `src/vamos/experiment/runner_output.py`: 15
- `src/vamos/engine/algorithm/spea2/spea2.py`: 15

Top offenders (after):
- `src/vamos/engine/algorithm/builders.py`: 21
- `src/vamos/ux/studio/app.py`: 21
- `src/vamos/engine/algorithm/moead/operators.py`: 20
- `src/vamos/operators/permutation.py`: 18
- `src/vamos/experiment/external/pymoo.py`: 17
- `src/vamos/experiment/runner_output.py`: 15
- `src/vamos/engine/algorithm/spea2/spea2.py`: 15
- `src/vamos/experiment/study/nsga2_diagnostics.py`: 15
- `src/vamos/engine/tuning/cli.py`: 15
- `src/vamos/engine/algorithm/smsemoa/operators.py`: 14

## CI changes
- Removed the `black --check .` step from `.github/workflows/ci.yml` (Ruff format gate is authoritative).

## Commands run (key)
- `.\.venv\Scripts\python.exe -m pytest -q` -> `301 passed, 2 skipped in 54.90s`
- `.\.venv\Scripts\mypy.exe --config-file pyproject.toml src/vamos` -> `Found 761 errors in 157 files`
- `.\.venv\Scripts\ruff.exe format src/vamos tests`
- Fast-fail gates:
  - `python -m pytest -q tests/architecture/test_layer_boundaries.py`
  - `python -m pytest -q tests/test_monolith_guard.py`
  - `python -m pytest -q tests/test_public_api_guard.py`
  - `python -m pytest -q tests/test_import_time_smoke.py`
  - `python -m pytest -q tests/test_optional_deps_policy.py`
  - `python -m pytest -q tests/test_logging_policy.py`
  - `python -m pytest -q tests/test_no_prints_in_library.py`
  - `python -m pytest -q tests/test_no_deprecation_shims.py`
  - `python -m pytest -q tests/test_agents_health_link.py`
  - `python -m pytest -q tests/architecture/test_ruff_gate.py`
  - `python -m pytest -q tests/architecture/test_ruff_format_gate.py`
  - `python -m pytest -q tests/architecture/test_py_typed_present.py`
  - `python -m pytest -q tests/architecture/test_mypy_error_budget.py`

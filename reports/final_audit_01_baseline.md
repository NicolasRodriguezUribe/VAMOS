# Final Audit Baseline 01

## Environment
- System Python: `Python 3.12.3`
- sys.executable: `C:\Users\nicor\AppData\Local\Programs\Python\Python312\python.exe`
- sys.version: `3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]`
- pip: `pip 24.0 (python 3.12)`
- ruff: not found on PATH
- mypy: not found on PATH; venv has `mypy 1.19.1 (compiled: yes)`
- Test interpreter: `.\.venv\Scripts\python.exe`

## Fast-Fail Health Gates
| Gate | Command | Result | Duration |
| --- | --- | --- | --- |
| Layer boundaries | `.\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_layer_boundaries.py` | PASS | 0.39s |
| Monolith guard | `.\.venv\Scripts\python.exe -m pytest -q tests/test_monolith_guard.py` | PASS | 0.65s |
| Public API guard | `.\.venv\Scripts\python.exe -m pytest -q tests/test_public_api_guard.py` | PASS | 0.03s |
| Import-time smoke | `.\.venv\Scripts\python.exe -m pytest -q tests/test_import_time_smoke.py` | PASS | 1.32s |
| Optional deps policy | `.\.venv\Scripts\python.exe -m pytest -q tests/test_optional_deps_policy.py` | PASS | 0.55s |
| Logging policy | `.\.venv\Scripts\python.exe -m pytest -q tests/test_logging_policy.py` | PASS | 0.51s |
| No prints in library | `.\.venv\Scripts\python.exe -m pytest -q tests/test_no_prints_in_library.py` | PASS | 0.50s |
| No deprecation shims | `.\.venv\Scripts\python.exe -m pytest -q tests/test_no_deprecation_shims.py` | PASS | 0.13s |
| AGENTS health link | `.\.venv\Scripts\python.exe -m pytest -q tests/test_agents_health_link.py` | PASS | 2.41s |

## Full Test Suite
- Command: `.\.venv\Scripts\python.exe -m pytest -q`
- Result: `296 passed, 2 skipped, 1 warning in 30.27s`

## Lint / Type / Build
### Ruff
- Command: `ruff --version`
- Result: not found on PATH

### Mypy
- Command: `.\.venv\Scripts\mypy.exe src/vamos`
- Result: FAIL (truncated)
```text
pyproject.toml: [mypy]: Unrecognized option: vamos.foundation.problem.* = {'strict': True}
pyproject.toml: [mypy]: Unrecognized option: vamos = {'engine': {'algorithm': {'config': {'strict': True}}}}
src/vamos/engine/algorithm/config/base.py:14: error: No overload variant of "asdict" matches argument type "_SerializableConfig"  [call-overload]
src/vamos/engine/algorithm/nsgaiii/helpers.py:17: error: Cannot find implementation or library stub for module named "vamos.foundation.problem.protocol"  [import-not-found]
src/vamos/engine/algorithm/nsgaiii/helpers.py:45: error: Need type annotation for "S"  [var-annotated]
...
Found 301 errors in 79 files (checked 271 source files)
```

### Build
- Command: `.\.venv\Scripts\python.exe -m build`
- Result: FAIL
```text
C:\Users\nicor\Desktop\VAMOS\.venv\Scripts\python.exe: No module named build
```

## Warnings
- `tests/ux/test_live_viz.py::test_live_pareto_plot_saves_file`: `FigureCanvasAgg is non-interactive, and thus cannot be shown`

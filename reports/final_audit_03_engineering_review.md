# Final Audit 03: Engineering Review

## A1) Architecture health gates currently present
- `tests/architecture/test_layer_boundaries.py`: enforces layer import rules (foundation/engine/ux/experiment).
- `tests/test_monolith_guard.py`: enforces file/function/class size thresholds (no allowlists).
- `tests/test_public_api_guard.py`: keeps `vamos/__init__.py` minimal and explicit.
- `tests/test_import_time_smoke.py`: ensures core facades import without heavyweight deps.
- `tests/test_optional_deps_policy.py`: enforces optional-dep imports by layer.
- `tests/test_logging_policy.py`: forbids `logging.basicConfig` in library modules.
- `tests/test_no_prints_in_library.py`: blocks `print()` in library layers (CLI/UI allowed).
- `tests/test_no_deprecation_shims.py`: forbids DeprecationWarning-style shims.
- `tests/test_agents_health_link.py`: requires AGENTS.md to reference architecture health doc.

Command outputs (sample):
```text
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_layer_boundaries.py
.                                                                        [100%]
1 passed in 0.26s
> .\.venv\Scripts\python.exe -m pytest -q tests/test_monolith_guard.py
.                                                                        [100%]
1 passed in 0.41s
> .\.venv\Scripts\python.exe -m pytest -q tests/test_import_time_smoke.py
.                                                                        [100%]
1 passed in 1.19s
```

## A2) Packaging & build readiness
- `pyproject.toml` has standard PEP 621 metadata; build backend is setuptools.
- Build backend module (`build`) not available in the current venv; build smoke fails until `build` is installed.

Command outputs:
```text
> .\.venv\Scripts\python.exe -m build
C:\Users\nicor\Desktop\VAMOS\.venv\Scripts\python.exe: No module named build
```

## A3) Lint/format readiness
- Ruff is not on PATH in the current environment; a dev extra should install it.
- Current config is minimal (E/F/W with a few ignores); keep this as the initial gate to avoid churn.

Command outputs:
```text
> ruff --version
ruff: The term 'ruff' is not recognized as a name of a cmdlet, function, script file, or executable program.
Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
```

## A4) Typing strategy
- Latest mypy run: Found 963 errors in 167 files (checked 271 source files).
- Proposed error-budget gate: fail if error count increases; allow gradual reductions PR by PR.

Top error categories (current):
| Code | Count |
| --- | ---: |
| `no-untyped-def` | 277 |
| `type-arg` | 209 |
| `no-untyped-call` | 82 |
| `arg-type` | 63 |
| `no-any-return` | 54 |
| `attr-defined` | 53 |
| `assignment` | 39 |
| `union-attr` | 33 |
| `import-untyped` | 29 |
| `misc` | 19 |

Top offender files (current):
| File | Count |
| --- | ---: |
| `src\vamos\experiment\wiring.py` | 35 |
| `src\vamos\foundation\constraints\dsl.py` | 34 |
| `src\vamos\engine\tuning\racing\bridge.py` | 30 |
| `src\vamos\experiment\external\jmetalpy.py` | 29 |
| `src\vamos\ux\studio\app.py` | 26 |
| `src\vamos\experiment\execution.py` | 26 |
| `src\vamos\foundation\kernel\moocore_backend.py` | 25 |
| `src\vamos\engine\algorithm\components\variation\pipeline.py` | 21 |
| `src\vamos\engine\algorithm\builders.py` | 21 |
| `src\vamos\engine\algorithm\moead\operators.py` | 20 |

Command outputs (truncated):
```text
> .\.venv\Scripts\mypy.exe --config-file pyproject.toml src/vamos
src\vamos\foundation\data\__init__.py:20: error: Argument 1 to "Path" has
incompatible type "Traversable"; expected "str | PathLike[str]"  [arg-type]
                return Path(alt)
                            ^~~
src\vamos\foundation\data\__init__.py:22: error: Argument 1 to "Path" has
incompatible type "Traversable"; expected "str | PathLike[str]"  [arg-type]
        return Path(path)
                    ^~~~
src\vamos\foundation\data\__init__.py:32: error: Argument 1 to "Path" has
incompatible type "Traversable"; expected "str | PathLike[str]"  [arg-type]
        return Path(path)
                    ^~~~
src\vamos\experiment\cli\types.py:9: error: Missing type parameters for generic
type "dict"  [type-arg]
        spec: dict
              ^
src\vamos\experiment\cli\types.py:10: error: Missing type parameters for
generic type "dict"  [type-arg]
        problem_overrides: dict
                           ^
src\vamos\experiment\cli\common.py:5: error: Function is missing a return type
annotation  [no-untyped-def]
    def _parse_probability_arg(parser, flag: str, raw, *, allow_expression...
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
src\vamos\experiment\cli\common.py:5: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def _parse_probability_arg(parser, flag: str, raw, *, allow_expression...
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
src\vamos\experiment\cli\common.py:27: error: Function is missing a return type
annotation  [no-untyped-def]
    def _parse_positive_float(parser, flag: str, raw, *, allow_zero: bool)...
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\cli\common.py:27: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def _parse_positive_float(parser, flag: str, raw, *, allow_zero: bool)...
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\cli\common.py:40: error: Function is missing a type
annotation  [no-untyped-def]
    def _normalize_operator_args(parser, args):
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\cli\common.py:79: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def collect_nsgaii_variation_args(args) -> dict:
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\cli\common.py:79: error: Missing type parameters for
generic type "dict"  [type-arg]
    def collect_nsgaii_variation_args(args) -> dict:
                                               ^
src\vamos\experiment\cli\common.py:97: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def _collect_generic_variation(args, prefix: str) -> dict:
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\cli\common.py:97: error: Missing type parameters for
generic type "dict"  [type-arg]
    def _collect_generic_variation(args, prefix: str) -> dict:
                                                         ^
src\vamos\engine\tuning\racing\schedule.py:7: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def build_schedule(
    ^~~~~~~~~~~~~~~~~~~
src\vamos\engine\tuning\racing\schedule.py:8: error: Missing type parameters
for generic type "Sequence"  [type-arg]
        instances: Sequence,
                   ^
src\vamos\engine\config\variation.py:9: error: Function is missing a type
annotation for one or more arguments  [no-untyped-def]
    def normalize_operator_tuple(spec) -> tuple[str, dict] | None:
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\engine\config\variation.py:9: error: Missing type parameters for
generic type "dict"  [type-arg]
    def normalize_operator_tuple(spec) -> tuple[str, dict] | None:
                                                     ^
src\vamos\engine\config\variation.py:28: error: Missing type parameters for
generic type "dict"  [type-arg]
    def normalize_variation_config(raw: dict | None) -> dict | None:
                                        ^
src\vamos\engine\config\variation.py:34: error: Missing type parameters for
generic type "dict"  [type-arg]
        normalized: dict = {}
                    ^
...
"import_pandas" in typed context  [no-untyped-call]
            pd = import_pandas()
                 ^~~~~~~~~~~~~~~
src\vamos\experiment\benchmark\report.py:149: error: Call to untyped function
"aggregate_metrics" in typed context  [no-untyped-call]
            tidy = self.aggregate_metrics()
                   ^~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\benchmark\report.py:196: error: Call to untyped function
"aggregate_metrics" in typed context  [no-untyped-call]
            tidy = self.aggregate_metrics()
                   ^~~~~~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\benchmark\cli.py:39: error: Returning Any from function
declared to return "dict[str, Any]"  [no-any-return]
            return json.load(fh)
            ^~~~~~~~~~~~~~~~~~~~
src\vamos\experiment\benchmark\cli.py:98: error: Call to untyped function
"aggregate_metrics" in typed context  [no-untyped-call]
        _ = report.aggregate_metrics()  # Writes tidy CSV
            ^~~~~~~~~~~~~~~~~~~~~~~~~~
Found 963 errors in 167 files (checked 271 source files)
```

## A5) Optional deps / import-time
- Import-time smoke + optional-deps policy currently pass; add gates for ruff/mypy/build to prevent regressions.
- Optional backends remain isolated in experiment/external or ux/studio; foundation/engine stay clean.

## A6) CI fast-fail ordering (recommended)
1) Architecture: layer boundaries, monolith guard, public API guard, import-time smoke
2) Policy checks: optional deps, logging policy, no-prints, no-deprecation shims, AGENTS link
3) Lint: ruff gate
4) Typing: mypy error budget
5) Build smoke: python -m build
6) Full test suite

## Updates after implementing missing gates
- Added dev tooling dependencies: `ruff>=0.6`, `mypy>=1.9`, `build>=1.2` in `pyproject.toml`.
- Added new fast-fail gates: `tests/architecture/test_ruff_gate.py`, `tests/architecture/test_mypy_error_budget.py`, `tests/architecture/test_build_smoke.py`.
- Added `tests/architecture/mypy_error_budget.json` with `max_errors: 963` to prevent regressions.
- CI fast-fail step now installs `.[dev]` and runs the new gates in order.
- Ruff cleanup fixes applied (unused imports, whitespace, one duplicate function, and an undefined variable) without changing behavior.

### New gates and how to run locally
```text
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_ruff_gate.py
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_mypy_error_budget.py
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_build_smoke.py
```

### Validation outputs
```text
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_ruff_gate.py
.                                                                        [100%]
1 passed in 0.12s
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_mypy_error_budget.py
.                                                                        [100%]
1 passed in 2.62s
> .\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_build_smoke.py
.                                                                        [100%]
1 passed in 22.77s
> .\.venv\Scripts\python.exe -m pytest -q
299 passed, 2 skipped, 1 warning in 41.72s
```

### Ruff configuration (current)
- Rules: `E`, `F`, `W` with ignores for `E402`, `E741`, `E501` to keep the gate low-churn.

### Mypy budget
- `max_errors`: 963 (baseline from latest run). Reduce over time; the gate fails on regressions.

### Runtime behavior
- No runtime behavior changes; edits are lint/typing hygiene only (unused imports, whitespace, and a duplicate function definition cleanup).
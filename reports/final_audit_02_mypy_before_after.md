# Final Audit 02: Mypy Before/After

## Changes applied
- Fixed mypy config to valid TOML overrides and added `mypy_path = ["src"]`.
- Updated `ProblemProtocol` import path to `vamos.foundation.problem.types` (typing-only).
- Added a type annotation for `S` in `nsgaiii/helpers.py` to satisfy strict var-annotated checks.
- Narrowed `asdict` input in `_SerializableConfig.to_dict` with a typing-only cast.

## Before (baseline from Final Audit 01)
```text
pyproject.toml: [mypy]: Unrecognized option: vamos.foundation.problem.* = {'strict': True}
pyproject.toml: [mypy]: Unrecognized option: vamos = {'engine': {'algorithm': {'config': {'strict': True}}}}
src/vamos/engine/algorithm/config/base.py:14: error: No overload variant of "asdict" matches argument type "_SerializableConfig"  [call-overload]
src/vamos/engine/algorithm/nsgaiii/helpers.py:17: error: Cannot find implementation or library stub for module named "vamos.foundation.problem.protocol"  [import-not-found]
src/vamos/engine/algorithm/nsgaiii/helpers.py:45: error: Need type annotation for "S"  [var-annotated]
...
Found 301 errors in 79 files (checked 271 source files)
```

## After (current mypy run)
- Command: `\.\.venv\Scripts\mypy.exe --config-file pyproject.toml src/vamos`
- Result: FAIL (Found 963 errors in 167 files (checked 271 source files))

Output (truncated):
```text
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
src\vamos\engine\config\variation.py:48: error: Missing type parameters for
generic type "dict"  [type-arg]
    ...sgaii_variation_config(encoding: str, overrides: dict | None) -> Dict[...
                                                        ^
src\vamos\engine\config\variation.py:60: error: Dict entry 0 has incompatible
type "str": "str"; expected "str": "float"  [dict-item]
                "mutation": ("bitflip", {"prob": "1/n"}),
                                         ^~~~~~~~~~~~~
src\vamos\engine\config\variation.py:65: error: Dict entry 0 has incompatible
type "str": "str"; expected "str": "float"  [dict-item]
                "mutation": ("reset", {"prob": "1/n"}),
                                       ^~~~~~~~~~~~~
src\vamos\engine\config\variation.py:70: error: Dict entry 0 has incompatible
type "str": "str"; expected "str": "float"  [dict-item]
                "mutation": ("mixed", {"prob": "1/n"}),
                                       ^~~~~~~~~~~~~
src\vamos\engine\config\variation.py:98: error: Missing type parameters for
generic type "dict"  [type-arg]
    def merge_variation_overrides(base: dict | None, override: dict | None...
                                        ^
src\vamos\adaptation\aos\portfolio.py:29: error: Need type annotation for
"_index" (hint: "_index: dict[<type>, <type>] = ...")  [var-annotated]
            self._index = {}
            ^~~~~~~~~~~
src\vamos\adaptation\aos\portfolio.py:62: error: Function is missing a type
annotation  [no-untyped-def]
        def __iter__(self):
        ^~~~~~~~~~~~~~~~~~~
src\vamos\adaptation\aos\logging.py:36: error: Argument 1 to "asdict" has
incompatible type "DataclassInstance | type[DataclassInstance]"; expected
"DataclassInstance"  [arg-type]
            return asdict(row)
                          ^~~
src\vamos\experiment\benchmark\report_utils.py:16: error: Function is missing a
return type annotation  [no-untyped-def]
    def import_pandas():
    ^~~~~~~~~~~~~~~~~~~~
src\vamos\engine\config\loader.py:29: error: Returning Any from function
declared to return "dict[str, Any]"  [no-any-return]
            return json.load(fh)
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

## Error counts
- Before: 301 errors in 79 files (config invalid).
- After: 963 errors in 167 files (config valid; broader checks).

## Top error categories (after)
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

## Top offender files (after)
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

## Notes
- The error count increased after config fixes because mypy now parses overrides correctly and checks more modules.
- Next step is to tackle high-volume `no-untyped-def` and `type-arg` issues in the top offender files.
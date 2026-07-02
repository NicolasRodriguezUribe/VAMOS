# Commands Used

| Command | Purpose | Status | Key Result |
|---|---|---:|---|
| `git status --short` | Check working tree before audit commands. | Succeeded | Initially clean. Later `.coverage` appeared after coverage run. |
| `bash -lc "find . -maxdepth 3 -type f \| sort"` | Required broad file inventory. | Succeeded | Produced large inventory including caches, results, site, source, tests, docs, paper artifacts. |
| `bash -lc "for d in src tests docs; do [ -d \"$d\" ] && find \"$d\" -type f; done \| sort"` | Required scoped file inventory attempt. | Failed | Shell quoting produced `syntax error: unexpected end of file`. |
| `python --version` | Record Python runtime. | Succeeded | `Python 3.14.2`. |
| `python -m pip show vamos` | Inspect installed `vamos` distribution. | Succeeded | Editable install reports `vamos 0.1.0`. |
| `bash -lc 'for d in src tests docs; do [ -d "$d" ] && find "$d" -type f; done \| sort'` | Retry scoped file inventory. | Succeeded | Returned no useful output due quoting/context behavior. |
| `python -m ruff check .` | Required repo-wide Ruff check. | Failed | 169 issues, mostly outside `src/vamos tests`; warning that `UP038` ignore is obsolete. |
| `python -m mypy src` | Required full source type check. | Failed | 8 errors in archive-related files. |
| `python -m pyright` | Required pyright check if available. | Failed | `No module named pyright`; tool unavailable. |
| `python -m vamos --help` | Required module CLI/help check. | Failed | No `vamos.__main__`; package cannot be directly executed. |
| `bash -lc "find src tests docs -type f \| sort"` | Required scoped file inventory. | Succeeded | Listed source, tests, docs, and also tracked/untracked cache files under those trees. |
| `python -m pytest -q` | Required test suite. | Succeeded | `1151 passed, 5 skipped, 19 warnings` in about 4.5 minutes. |
| `python -m pytest --cov=src --cov-report=term-missing` | Required coverage run. | Succeeded | `1151 passed, 5 skipped`; total coverage `69%`. |
| `Get-Content pyproject.toml` | Inspect packaging metadata and tool configuration. | Succeeded | Confirmed project name/version, extras, script entry point, mypy/Ruff config. |
| `python -c "import importlib.metadata as md; ..."` | Inspect installed console scripts. | Succeeded | Found `vamos = vamos.experiment.cli.main:main` twice in current environment. |
| `vamos --help` | Check equivalent CLI help command. | Succeeded | Console script printed argparse help. |
| `python -m pip show vamos-optimization` | Inspect installed package metadata. | Succeeded | Editable install reports `vamos-optimization 1.0.0`. |
| `python -m coverage report --sort=Cover` | Sort coverage by lowest-covered modules. | Succeeded | Confirmed 0% and low-coverage modules; total 69%. |
| `rg --files -g '*.py' src tests docs` | Enumerate Python source/test files quickly. | Succeeded | Listed Python files under source and tests. |
| `bash -lc "nl -ba src/vamos/__init__.py \| sed -n '1,220p'"` | Inspect package facade lines. | Succeeded | Confirmed public exports and lazy `problems`. |
| `bash -lc "nl -ba src/vamos/api.py \| sed -n '1,220p'"` | Inspect user-facing API facade. | Succeeded | Confirmed stable entrypoint exports. |
| `bash -lc "nl -ba src/vamos/algorithms.py \| sed -n '1,220p'"` | Inspect algorithm facade. | Succeeded | Confirmed config exports and discovery helpers. |
| `bash -lc "nl -ba src/vamos/foundation/problem/base.py \| sed -n '1,240p'"` | Inspect custom problem base protocol. | Succeeded | Confirmed batched objective/constraint validation. |
| `bash -lc "nl -ba docs/guide/getting-started.md \| sed -n '1,220p'"` | Inspect onboarding docs. | Succeeded | Confirmed install, CLI, optimize, make_problem guidance. |
| `bash -lc "nl -ba src/vamos/experiment/unified.py \| sed -n '1,430p'"` | Inspect `optimize` API and resolved metadata. | Succeeded | Confirmed auto defaults and result metadata. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/config/base.py \| sed -n '1,340p'"` | Inspect config builder infrastructure. | Succeeded | Confirmed shared builder mixins and validation. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/config/moead.py \| sed -n '1,220p'"` | Inspect active-file MOEA/D config. | Succeeded | Confirmed defaults and builder fields. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/registry.py \| sed -n '1,260p'"` | Inspect algorithm registry and plugin loading. | Succeeded | Confirmed built-ins and `vamos.algorithms` plugin group. |
| `bash -lc "nl -ba src/vamos/experiment/cli/main.py \| sed -n '1,260p'"` | Inspect CLI dispatch. | Succeeded | Confirmed subcommands and standard CLI path. |
| `bash -lc "nl -ba README.md \| sed -n '1,240p'"` | Inspect README claims and onboarding. | Succeeded | Confirmed features, install, quickstart, tuning guidance. |
| `bash -lc "nl -ba src/vamos/foundation/kernel/numpy_backend.py \| sed -n '1,430p'"` | Inspect core vectorized kernels. | Succeeded | Confirmed dense/blocked nondominated sort and tournament sampling. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/nsgaii/helpers.py \| sed -n '1,280p'"` | Inspect NSGA-II helpers and selection/survival paths. | Succeeded | Confirmed selection and crowding helpers. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/moead/moead.py \| sed -n '1,280p'"` | Inspect MOEA/D loop and batching. | Succeeded | Confirmed ask/tell loop and batch parent sampling. |
| `bash -lc "nl -ba src/vamos/engine/archive/bounded_archive.py \| sed -n '1,360p'"` | Inspect archive pruning paths. | Succeeded | Confirmed HV, crowding, KNN, maxmin, ref-dir pruning paths. |
| `bash -lc "nl -ba tests/architecture/test_layer_boundaries.py \| sed -n '1,220p'"` | Inspect architecture layer guard. | Succeeded | Confirmed allowed layer import policy. |
| `bash -lc "nl -ba tests/architecture/test_mypy_error_budget.py \| sed -n '1,220p'"` | Inspect mypy CI scope test. | Succeeded | Confirmed CI scope assertions. |
| `bash -lc "nl -ba src/vamos/foundation/kernel/numpy_backend.py \| sed -n '425,520p'"` | Inspect NSGA-II survival and HV delegation. | Succeeded | Confirmed merge/rank/select survival and HV call. |
| `bash -lc "nl -ba src/vamos/foundation/eval/backends.py \| sed -n '1,290p'"` | Inspect evaluation backends. | Succeeded | Confirmed serial, multiprocessing, Dask, and fallback behavior. |
| `bash -lc "nl -ba src/vamos/foundation/problem/builder.py \| sed -n '1,340p'"` | Inspect function problem builder. | Succeeded | Confirmed scalar row-wise adapter and validation. |
| `bash -lc "nl -ba src/vamos/foundation/quality_indicators/hypervolume.py \| sed -n '1,330p'"` | Inspect hypervolume implementation and fallback. | Succeeded | Confirmed generic contribution loop. |
| `bash -lc "nl -ba src/vamos/experiment/optimization_result/model.py \| sed -n '1,210p'"` | Inspect result API. | Succeeded | Confirmed `front`, `best`, `top_k`, defaults explanation. |
| `bash -lc "nl -ba docs/topics/engineering_audit.md \| sed -n '1,240p'"` | Inspect historical engineering audit context. | Succeeded | Treated as historical context, not proof. |
| `bash -lc "nl -ba pyproject.toml \| sed -n '1,260p'"` | Inspect full packaging/tool config with line numbers. | Succeeded | Confirmed extras, scripts, mypy, pytest, Ruff config. |
| `bash -lc "nl -ba src/vamos/foundation/version.py \| sed -n '1,80p'"` | Inspect runtime version. | Succeeded | Confirmed `__version__ = "0.1.0"`. |
| `bash -lc "nl -ba .github/workflows/ci.yml \| sed -n '1,260p'"` | Inspect CI gates. | Succeeded | Confirmed Ruff, mypy subset, tests, docs, wheel smoke. |
| `bash -lc "nl -ba tests/architecture/test_ruff_gate.py \| sed -n '1,220p'"` | Inspect Ruff lint gate. | Succeeded | Confirmed `ruff check src/vamos tests`. |
| `bash -lc "nl -ba tests/architecture/test_ruff_format_gate.py \| sed -n '1,220p'"` | Inspect Ruff format budget gate. | Succeeded | Confirmed budget-based format check. |
| `python -m ruff check src/vamos tests` | Check CI-style Ruff scope. | Succeeded | All checks passed. |
| `python -m ruff format --check src/vamos tests` | Check formatting scope. | Failed | 56 files would be reformatted. |
| `python -m mypy --config-file pyproject.toml src/vamos/engine/algorithm/config src/vamos/engine/algorithm/registry.py src/vamos/engine/config/spec.py src/vamos/foundation/eval src/vamos/experiment/cli/common.py src/vamos/experiment/optimization_result src/vamos/experiment/unified.py` | Check CI-style mypy scope. | Failed | Same 8 archive-related errors because imports pull archive modules. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/components/archive.py \| sed -n '1,120p'"` | Inspect mypy archive private access. | Succeeded | Confirmed `_subset_selection._moocore` use. |
| `bash -lc "nl -ba src/vamos/engine/algorithm/components/subset_selection.py \| sed -n '1,80p'"` | Inspect optional MooCore variable. | Succeeded | Confirmed `_moocore` private optional global. |
| `bash -lc "nl -ba src/vamos/engine/archive/factory.py \| sed -n '1,210p'"` | Inspect archive type aliases. | Succeeded | Confirmed runtime `Any` alias pattern. |
| `bash -lc "nl -ba src/vamos/engine/hooks/hv_archive_hooks.py \| sed -n '110,155p'"` | Inspect hook archive annotation. | Succeeded | Confirmed `ResultArchiveManager | None` annotation. |
| `git status --short` | Check generated artifacts after coverage. | Succeeded | `.coverage` untracked. |
| `Get-Content tests/architecture/ruff_format_budget.json` | Inspect format debt budget. | Succeeded | Budget allows up to 100 files. |
| `Get-Content tests/architecture/mypy_error_budget.json` | Inspect mypy debt budget. | Succeeded | Budget allows up to 72 errors. |
| `rg -n "version\|__version__\|vamos-optimization\|Development Status\|python-version\|python_version" ...` | Find version/package references. | Succeeded | Found pyproject 1.0.0 and runtime/citation 0.1.0 references. |
| `bash -lc "nl -ba README.md \| sed -n '320,370p'"` | Inspect README tuning/citation area. | Succeeded | Confirmed citation version 0.1.0. |
| `bash -lc "nl -ba tests/architecture/mypy_error_budget.json \| sed -n '1,80p'"` | Capture mypy budget lines. | Succeeded | Confirmed max_errors 72. |
| `bash -lc "nl -ba tests/architecture/ruff_format_budget.json \| sed -n '1,80p'"` | Capture format budget lines. | Succeeded | Confirmed max_files_to_reformat 100. |
| `Get-Content CITATION.cff` | Inspect citation metadata. | Succeeded | Confirmed version 0.1.0 and release date. |
| `rg -n "CITATION\|citation\|version:\|version =" CITATION.cff README.md pyproject.toml src/vamos/foundation/version.py` | Find citation/version references. | Succeeded | Confirmed mismatched versions. |
| `python -c "import vamos; print(vamos.__version__)"` | Check runtime package version. | Succeeded | Printed `0.1.0`. |
| `python -c "import importlib.metadata as md; print(md.version('vamos-optimization'))"` | Check distribution version. | Succeeded | Printed `1.0.0`. |
| `Test-Path docs/audit` | Check target audit directory. | Succeeded | Returned `False`. |
| `bash -lc "nl -ba pyproject.toml \| sed -n '1,180p'"` | Re-check packaging refs before writing artifacts. | Succeeded | Confirmed line references. |
| `bash -lc "nl -ba src/vamos/foundation/version.py \| sed -n '1,40p'"` | Re-check runtime version refs. | Succeeded | Confirmed line references. |
| `bash -lc "nl -ba CITATION.cff \| sed -n '1,30p'"` | Re-check citation refs. | Succeeded | Confirmed line references. |
| `bash -lc "nl -ba README.md \| sed -n '337,352p'"` | Re-check README citation refs. | Succeeded | Confirmed line references. |
| `New-Item -ItemType Directory -Force docs/audit \| Out-Null` | Create allowed audit artifact directory. | Succeeded | Created `docs/audit`. |
| `bash -lc "nl -ba src/vamos/experiment/benchmark/cli.py \| sed -n '1,120p'"` | Inspect benchmark CLI low-coverage module. | Succeeded | Confirmed CLI parser/main lines. |
| `bash -lc "nl -ba src/vamos/experiment/profiler/runner.py \| sed -n '1,170p'"` | Inspect profiler low-coverage module. | Succeeded | Confirmed run_profile behavior. |
| `bash -lc "nl -ba src/vamos/ux/panel/pages/problem_builder.py \| sed -n '1,80p'"` | Inspect Panel problem-builder low-coverage module. | Succeeded | Confirmed UI state defaults. |
| `bash -lc "nl -ba src/vamos_contrib/interop/pymoo.py \| sed -n '1,80p'"` | Inspect pymoo interop low-coverage module. | Succeeded | Confirmed adapter behavior. |
| Python CSV validation via `csv.DictReader` | Validate `docs/audit/findings.csv` schema, row count, severities, efforts, and path/line ranges. | Succeeded | `CSV OK: 10 findings`. |
| Python markdown reference validation | Validate every markdown reference matching `path:Lx-Ly` exists and line ranges are in bounds. | Succeeded | `Line refs OK: 36 refs`. |
| `git status --short` | Confirm final modified/untracked files. | Succeeded | Only `.coverage` and `docs/audit/` were shown. |
| Python CSV re-validation via `csv.DictReader` | Re-check CSV after appending validation commands to this log. | Succeeded | `CSV still OK`. |
| `git status --short` | Final working tree check after artifact/log updates. | Succeeded | Only `.coverage` and `docs/audit/` were shown. |

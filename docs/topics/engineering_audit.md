# VAMOS Software Engineering Best Practices Audit

## Executive summary
- Operators are split into `src/vamos/operators/impl` (implementations) and `src/vamos/operators/policies` (algorithm wiring), and an architecture boundary test now guards layer inversions; foundation->engine and engine->ux edges are removed.
- Import graph is cleaner, but `experiment.runner` and `experiment.study.runner` form a static cycle; coupling is concentrated in experiment->engine/ux and a small foundation->operators dependency for kernels.
- Experiment orchestration modules (runner/quick/CLI) remain large and mix execution, IO, and presentation, creating monolith risk.
- Algorithm classes are large but mostly cohesive; variation pipeline and config classes remain heavy and will be hard to evolve without stronger interfaces.
- Public API remains broad; optimize/runner now live under experiment while foundation core API is primitives-only.
- Logging is still inconsistent: many `print` statements exist in experiment runtime modules instead of structured logging.
- Config specs are flexible but lack explicit schema/versioning and centralized validation outside the CLI.
- Highest leverage next steps: split experiment runner/CLI wiring, separate result presentation from optimize/quick, and formalize config schema.

## Repository architecture map
Key directories (concise):
```
src/vamos/
  __init__.py, api.py              # Public API facade and reexports
  foundation/                      # Core types, kernels, problems, metrics, constraints, eval
  operators/                       # Operator packages (impl + policies)
  engine/                          # Algorithms, configs, tuning, hyperheuristics
  experiment/                      # CLI, diagnostics, benchmarks, study runner, quick API
  ux/                              # Analysis, visualization, studio app
  hooks/, adaptation/, archive/    # Extensions and runtime hooks
  io/, monitoring/                 # Auxiliary IO/monitoring (thin in current tree)

tests/                             # Unit and integration tests
examples/, notebooks/, experiments/# Research scripts and demos
docs/                              # MkDocs site + dev guides
```
Entry points (from `pyproject.toml`):
- `vamos` -> `src/vamos/experiment/cli/main.py`
- `vamos-self-check` -> `src/vamos/experiment/diagnostics/self_check.py`
- `vamos-benchmark` -> `src/vamos/experiment/benchmark/cli.py`
- `vamos-studio` -> `src/vamos/ux/studio/app.py`
- `vamos-zoo` -> `src/vamos/experiment/zoo/cli.py`
- `vamos-tune` -> `src/vamos/engine/tuning/cli.py`

Primary public APIs:
- `src/vamos/__init__.py` reexports high-level functions, configs, plotting, and problem registries.
- `src/vamos/api.py` is the main programmatic entrypoint; `src/vamos/foundation/core/api.py` exposes primitives only, while `src/vamos/engine/api.py` and `src/vamos/ux/api.py` provide focused facades.

Core layers and responsibilities (observed):
- foundation: problems/registries, kernels/backends, metrics, constraints, eval backends, core types.
- operators: operator implementations (impl) and algorithm wiring (policies).
- engine: algorithm implementations, algorithm configs, tuning, hyperheuristics.
- experiment: runner/optimize execution, CLI/config parsing, benchmark/study orchestration, diagnostics, quick wrappers.
- ux: analysis, stats, visualization, streamlit Studio UI.

## Monolith findings
Top 20 largest Python files under `src/vamos/` by non-blank LOC (static scan):

| File | LOC | Responsibilities | Monolith risk |
| --- | ---: | --- | --- |
| `src/vamos/engine/algorithm/components/base.py` | 574 | Base state, termination, population setup, archives, HV tracking, hooks for live viz + genealogy, results | High: many algorithm concerns + instrumentation coupling |
| `src/vamos/foundation/problem/registry/specs.py` | 541 | Problem registry data and factories for all benchmark families | Medium: huge registry; consider splitting by family |
| `src/vamos/experiment/runner.py` | 536 | Orchestrates runs, builds algorithms, HV, hooks, plotting, persistence | High: execution + IO + plotting + config overrides |
| `src/vamos/foundation/core/external/base.py` | 510 | External baseline adapters (pymoo/jmetalpy/pygmo), wrappers, printing | High: integration + CLI output + optional deps |
| `src/vamos/experiment/cli/parser.py` | 491 | CLI parsing, config loading, validation, defaults for algorithms | High: parsing + config semantics + validation |
| `src/vamos/operators/impl/permutation.py` | 428 | All permutation operators and optional numba paths | Medium: large single-purpose operator library |
| `src/vamos/operators/impl/real/crossover.py` | 379 | Many real-valued crossover operators | Medium: large operator module |
| `src/vamos/experiment/optimize.py` | 370 | Optimize API, result helpers, plotting, saving | Medium: core + presentation/IO |
| `src/vamos/foundation/kernel/moocore_backend.py` | 332 | MooCore kernel + HV + archives + tournament selection | Medium: kernel + metrics/selection coupling |
| `src/vamos/operators/impl/real/mutation.py` | 327 | Many real-valued mutation operators | Medium: large operator module |
| `src/vamos/engine/algorithm/nsgaiii/nsgaiii.py` | 324 | NSGA-III algorithm loop and helpers | Medium: large but cohesive |
| `src/vamos/engine/algorithm/smsemoa/smsemoa.py` | 322 | SMS-EMOA algorithm loop and helpers | Medium: large but cohesive |
| `src/vamos/engine/algorithm/smpso/smpso.py` | 311 | SMPSO algorithm loop and helpers | Medium: large but cohesive |
| `src/vamos/ux/studio/app.py` | 310 | Streamlit UI, data loading, MCDM, focused runs, plots | High: UI + analysis + execution mixed |
| `src/vamos/engine/algorithm/nsgaii/nsgaii.py` | 307 | NSGA-II algorithm loop, AOS integration | Medium: large but cohesive |
| `src/vamos/engine/algorithm/ibea/ibea.py` | 298 | IBEA algorithm loop and helpers | Medium: large but cohesive |
| `src/vamos/engine/algorithm/nsgaiii/helpers.py` | 291 | NSGA-III helper routines | Low/Medium: helper module scale |
| `src/vamos/engine/algorithm/nsgaii/state.py` | 291 | NSGA-II state, result building, genealogy/archives | Medium: algorithm + instrumentation coupling |
| `src/vamos/engine/algorithm/spea2/spea2.py` | 284 | SPEA2 algorithm loop and helpers | Medium: large but cohesive |

God-class candidates (top by LOC/method count):
- `src/vamos/engine/algorithm/nsgaiii/nsgaiii.py` `NSGAIII` (~352 LOC, 10 methods)
- `src/vamos/engine/algorithm/smsemoa/smsemoa.py` `SMSEMOA` (~347 LOC, 10 methods)
- `src/vamos/engine/algorithm/smpso/smpso.py` `SMPSO` (~332 LOC, 8 methods)
- `src/vamos/engine/algorithm/ibea/ibea.py` `IBEA` (~313 LOC, 11 methods)
- `src/vamos/engine/algorithm/nsgaii/nsgaii.py` `NSGAII` (~303 LOC, 9 methods)
- `src/vamos/engine/algorithm/spea2/spea2.py` `SPEA2` (~302 LOC, 8 methods)
- `src/vamos/engine/tuning/racing/core.py` `RacingTuner` (~264 LOC, 8 methods)
- `src/vamos/engine/algorithm/components/variation/pipeline.py` `VariationPipeline` (~251 LOC, 7 methods)
- `src/vamos/engine/algorithm/config/nsgaii.py` `NSGAIIConfig` (~250 LOC, 21 methods)
- `src/vamos/experiment/context.py` `Experiment` (~248 LOC, 10 methods)

God-function candidates (top by LOC):
- `src/vamos/experiment/cli/parser.py` `parse_args` (~450 LOC)
- `src/vamos/ux/studio/app.py` `main` (~236 LOC)
- `src/vamos/experiment/runner.py` `run_single` (~175 LOC)
- `src/vamos/experiment/runner.py` `execute_problem_suite` (~148 LOC)
- `src/vamos/engine/algorithm/ibea/initialization.py` `initialize_ibea_run` (~137 LOC)

Cross-cutting tangling hotspots:
- `src/vamos/experiment/runner.py`: execution + config override + plotting + persistence + CLI hints.
- `src/vamos/experiment/cli/parser.py`: parsing + default merging + validation + algorithm-specific operator config.
- `src/vamos/experiment/quick.py`: API + config building + plotting + saving.
- `src/vamos/ux/studio/app.py`: UI + data loading + optimization runs + analysis/plots.
- `src/vamos/experiment/optimize.py`: core optimize + result presentation/IO.

Utility modules that are growing:
- `src/vamos/engine/algorithm/components/utils.py`
- `src/vamos/engine/algorithm/components/variation/helpers.py`
- `src/vamos/operators/impl/real/utils.py`
- `src/vamos/foundation/constraints/utils.py`
- `src/vamos/experiment/runner_utils.py`
- `src/vamos/foundation/core/io_utils.py`

## Dependency and import-graph findings
Static import graph summary:
- Internal modules scanned: 276
- Internal import edges: 563
- Module-level cycles detected: 1 (`experiment.runner` <-> `experiment.study.runner` via an import inside `run_study`)

Cross-layer edges by count (top-level package segment):
- engine -> foundation: 71
- experiment -> foundation: 41
- engine -> operators: 33
- engine -> hooks: 12
- experiment -> engine: 9
- experiment -> ux: 4
- foundation -> operators: 2
- ux -> engine: 1
- ux -> foundation: 1

Examples of risky couplings:
- `src/vamos/experiment/runner.py` imports `vamos.engine.algorithm.factory` and `vamos.engine.config.*` (experiment -> engine).
- `src/vamos/experiment/runner.py` imports `vamos.ux.visualization.plotting` and `LiveParetoPlot` (experiment -> ux).
- `src/vamos/experiment/study/runner.py` imports `run_single` from `vamos.experiment.runner`, while `run_study` in runner imports `StudyRunner` (static cycle).
- `src/vamos/foundation/kernel/numpy_backend.py` imports `vamos.operators.impl.real.*` (foundation -> operators; shared package, still a tight dependency).
- `src/vamos/ux/studio/app.py` imports `vamos.engine.algorithm.config.NSGAIIConfig` (ux -> engine).

There is a single static cycle between `experiment.runner` and `experiment.study.runner`; otherwise, foundation->engine/ux and engine->ux edges are cleared, improving backend modularity and UX replaceability.

## Engineering hygiene findings
Packaging:
- `pyproject.toml` defines core dependencies and a comprehensive set of extras; entry points are clearly declared.
- `src/vamos/py.typed` is present, enabling type-checking for consumers.
- Package data for reference fronts and weights is declared in `pyproject.toml`.

Tests:
- 82 test files under `tests/`, with markers for optional features.
- Integration tests include minimal-import checks and packaging-data checks.
- Architecture boundary test (`tests/architecture/test_layer_boundaries.py`) enforces zero foundation->engine/ux and engine->ux imports.

CI/tooling:
- GitHub Actions run full tests, lint, format, mypy, and docs builds; also run minimal install smoke tests.
- Pre-commit config includes ruff/black/mypy and basic hygiene hooks.

Typing:
- mypy config is strict for selected modules; other areas are typed but not enforced.
- Some public APIs accept `Any` and free-form dicts (e.g., config specs) without schema validation.

Docs:
- MkDocs site with broad coverage and dev guides exists.
- No dedicated architecture/layering document codifying dependency rules.

Logging and error handling:
- A custom exception hierarchy exists.
- Many experiment/runtime modules use `print` (runner, optimize, quick API, tuning), which is brittle for library usage.

Configuration:
- YAML/JSON specs are supported via `src/vamos/engine/config/loader.py` with manual validation in CLI.
- No schema/versioning for config files, and no centralized validation outside the CLI.

Performance boundaries:
- Multiple kernel backends (NumPy, Numba, MooCore) exist; kernels depend on the shared operators package but not engine, improving backend modularity.

## Prioritized recommendations (ranked)
1) High - Split the experiment runner/CLI pipeline and remove the `experiment.runner` <-> `experiment.study.runner` cycle.
Evidence: `src/vamos/experiment/runner.py`, `src/vamos/experiment/study/runner.py`, `src/vamos/experiment/cli/parser.py`.
Refactor plan: extract shared execution logic (e.g., `src/vamos/experiment/execution.py`), make `StudyRunner` depend on an execution interface, and move CLI defaults/validation into dedicated `cli/args.py` and `cli/validation.py`; eliminate mutual imports via dependency inversion.
Suggested tests: add a lightweight import-cycle check for experiment modules; unit tests for config override resolution; integration tests that CLI parsing matches programmatic calls.

2) High - Separate result presentation (plotting, saving, printing) from core API results.
Evidence: `src/vamos/experiment/optimization_result.py`.
Refactor plan: keep `OptimizationResult` as a data container; move `plot`, `save`, and `summary` to `src/vamos/ux/` and wire them from CLI/UX.
Suggested tests: unit tests for output helpers; integration tests for CLI output parity.

3) Medium - Split external baseline integrations into per-library modules.
Evidence: `src/vamos/foundation/core/external/base.py`.
Refactor plan: move pymoo/jmetalpy/pygmo adapters to `src/vamos/experiment/benchmark/external/` modules; make `resolve_external_algorithm` a registry in that package; keep imports lazy and outputs structured (no prints).
Suggested tests: dependency-missing tests that verify informative error messages; functional tests for one baseline when optional deps are available.

4) Medium - Break up the Streamlit Studio app into UI and service layers.
Evidence: `src/vamos/ux/studio/app.py`.
Refactor plan: keep UI layout in `app.py`; move data loading, focused runs, and dynamics capture into `src/vamos/ux/studio/services.py` and reuse existing `data.py`.
Suggested tests: unit tests for data loading and focused-optimization logic (pure functions) without UI.

5) Medium - Replace `print` usage in experiment/runtime modules with structured logging.
Evidence: widespread `print` in `src/vamos/experiment/runner.py`, `src/vamos/experiment/optimize.py`, `src/vamos/experiment/quick.py`, `src/vamos/engine/tuning/*`.
Refactor plan: introduce module-level loggers and a consistent logging policy; keep CLI responsible for user-facing console output; add `--quiet` or `--verbose` flags for CLI.
Suggested tests: verify logging is emitted for key events and that CLI still prints expected summaries.

6) Medium - Formalize config schema and versioning.
Evidence: `src/vamos/engine/config/loader.py`, `src/vamos/experiment/cli/parser.py`.
Refactor plan: define an `ExperimentSpec` dataclass or pydantic model; add schema validation and a version field; centralize defaults and merge logic.
Suggested tests: schema validation tests for typical configs; regression tests for YAML/JSON compatibility.

7) Low - Split large operator modules by operator family.
Evidence: `src/vamos/operators/impl/permutation.py`, `src/vamos/operators/impl/real/crossover.py`, `src/vamos/operators/impl/real/mutation.py`.
Refactor plan: move each operator family into dedicated files and provide registries for lookup; keep public imports stable.
Suggested tests: existing operator tests should continue to pass; add import performance checks if needed.

## No Monolith guidance
Recommended thresholds:
- Core modules: <= 400 LOC and <= 3 major responsibilities.
- CLI/UI modules: <= 250-300 LOC, prefer orchestration only.
- Algorithm classes: <= 350 LOC, push setup/metrics into helpers or base classes.

Layering rules (proposed):
- foundation must not import engine or ux.
- operators.impl is shared; foundation/engine/experiment/ux may depend on it, but it must not depend on engine/ux.
- operators.policies may depend on engine for algorithm wiring, but should avoid ux.
- engine may depend on foundation and hooks, but not on ux.
- experiment and ux may depend on foundation and engine.
- avoid experiment <-> ux mutual imports; prefer hooks and experiment-driven wiring for optional visualization.
- CLI modules should not be imported by foundation/engine.

Dependency inversion suggestions:
- Use Protocols/ABCs for live visualization, genealogy, and external hooks in the neutral `src/vamos/hooks/` package.
- Use registries for algorithms, operators, and backends to avoid direct imports.
- Prefer passing dependency objects (kernel, evaluator, visualizer) instead of importing their concrete implementations.

## Architecture and layering document outline (to add under docs/)
- Purpose and guiding principles (research-friendly, modular, extensible)
- Layer definitions and allowed dependencies
- Public APIs and stability guarantees
- Plugin/registry patterns (algorithms, kernels, operators)
- Data contracts (results layout, metadata, config schema)
- Performance boundaries (where vectorization/JIT lives)
- Optional dependencies and feature flags
- Testing strategy by layer (unit, integration, optional backends)

## Next 2 PRs (high leverage, well scoped)
1) PR: Split experiment runner/CLI responsibilities and break the `experiment.runner` <-> `experiment.study.runner` cycle.
Scope: extract shared execution logic into `src/vamos/experiment/execution.py`, move CLI defaults/validation into dedicated modules, and add a small import-cycle guard for experiment modules.

2) PR: Separate output/presentation from optimize/quick and standardize logging.
Scope: move plot/save/summary helpers into `src/vamos/experiment/output.py` or `src/vamos/ux/`, replace runtime `print` calls with structured logging, and add tests covering output helpers.

## Appendix: Commands run
- `Get-ChildItem -Force` (list repo root)
- `Get-ChildItem -Directory src` (list src)
- `Get-ChildItem -Directory src\vamos` (list package root)
- `Get-ChildItem -Directory tests` (list tests)
- `Get-ChildItem -Directory docs` (list docs)
- `Get-Content pyproject.toml` (packaging, entry points, tools)
- `Get-Content src\vamos\__init__.py` (public API reexports)
- `@'...python script...'@ | python -` (first attempt: top 20 LOC, ended with stray `PY` name error)
- `@'...python script...'@ | python -` (top 20 LOC scan, successful)
- `@'...python script...'@ | python -` (top classes/functions by LOC)
- `@'...python script...'@ | python -` (import graph and cross-layer edges)
- `rg -n "from\s+vamos\s+import\s+\*"` (wildcard import check)
- `rg --files -g "*utils*.py" -g "*helpers*.py" -g "*common*.py"` (utility module inventory)
- `Get-Content src\vamos\api.py` (public API)
- `Get-ChildItem src\vamos\experiment\cli` (CLI module listing)
- `Get-Content README.md` (repo overview)
- `Get-ChildItem -Recurse .github` (CI/workflows listing)
- `Get-Content .github\workflows\ci.yml` (CI details)
- `Get-Content .github\workflows\upload_pypi.yml` (release workflow)
- `rg -n "\bprint\(" src\vamos` (print usage)
- `Get-ChildItem -Directory src\vamos\engine` (engine tree)
- `Get-ChildItem -Directory src\vamos\foundation` (foundation tree)
- `Get-ChildItem -Directory src\vamos\experiment` (experiment tree)
- `Get-ChildItem -Directory src\vamos\ux` (ux tree)
- `Get-ChildItem -Directory src\vamos\adaptation,src\vamos\archive,src\vamos\hooks,src\vamos\io,src\vamos\monitoring` (aux package listing)
- `Get-ChildItem src\vamos\hooks` (hooks files)
- `rg --files tests -g "*.py" | Measure-Object` (test file count)
- `Get-Content tests\integration\test_minimal_imports.py` (minimal import tests)
- `rg --files -g "py.typed"` (typing marker check)
- `rg -n "vamos\.engine" src\vamos\foundation` (foundation -> engine imports)
- `Get-Content src\vamos\experiment\quick.py` (quick API)
- `Get-Content src\vamos\engine\algorithm\components\base.py` (base algorithm components)
- `Get-Content src\vamos\foundation\problem\registry\specs.py` (problem registry)
- `Get-Content src\vamos\foundation\core\external\base.py` (external baselines)
- `Get-Content src\vamos\experiment\cli\parser.py` (CLI parser)
- `Get-Content src\vamos\experiment\runner.py` (runner)
- `Get-Content src\vamos\experiment\optimize.py` (optimize API)
- `Get-Content src\vamos\foundation\kernel\moocore_backend.py` (MooCore backend)
- `Get-Content src\vamos\ux\studio\app.py` (studio app)
- `Get-Content src\vamos\operators\impl\permutation.py` (permutation operators)
- `Get-Content src\vamos\engine\algorithm\nsgaii\state.py` (NSGA-II state)
- `Get-Content src\vamos\engine\algorithm\nsgaii\helpers.py` (NSGA-II helpers)
- `Get-Content src\vamos\foundation\eval\backends.py` (eval backends)
- `Get-Content src\vamos\foundation\kernel\numpy_backend.py` (NumPy backend)
- `Get-Content src\vamos\engine\algorithm\components\hypervolume.py` (hypervolume utilities)
- `Get-Content src\vamos\foundation\exceptions.py` (exception hierarchy)
- `Get-Content src\vamos\engine\algorithm\nsgaii\nsgaii.py` (NSGA-II core)
- `Get-Content src\vamos\operators\impl\real\crossover.py` (real crossover operators)
- `Get-Content src\vamos\foundation\problem\registry\__init__.py` (problem registry exports)
- `Get-Content src\vamos\foundation\core\api.py` (foundation API)
- `Get-ChildItem docs\dev` (dev docs listing)
- `Get-ChildItem docs\experiment` (experiment docs listing)
- `Get-Content mkdocs.yml` (docs config)
- `Get-Content .pre-commit-config.yaml` (pre-commit config)
- `Get-Content src\vamos\engine\config\loader.py` (config loader)
- `Get-Content experiments\scripts\collect_hv_archive_metrics.py` (metrics script)
- `@'...python script...'@ | .\.venv\Scripts\python.exe -` (refresh LOC/classes/functions/import graph)
- `rg --files tests -g "*.py" | Measure-Object` (test file count refresh)
- `rg -n "ux\.visualization|ux\.analysis|ux\.studio" src\vamos\experiment\runner.py` (experiment -> ux imports)
- `rg -n "vamos\.engine" src\vamos\ux` (ux -> engine imports)
- `rg -n "foundation\.core\.(runner|optimize|runner_utils)" src tests docs examples experiments` (verify old import paths removed)

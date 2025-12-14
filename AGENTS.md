# AGENTS.md

> For a user-facing overview of VAMOS (features, install, CLI examples), see `README.md`.
> For contributor workflows and concrete tasks, see `AGENTS_tasks.md`.
> For ready-to-paste prompts for code assistants, see `AGENTS_codex_prompts.md`.

Guidance for AI coding agents (Codex, GPT, Copilot, etc.) working on the **VAMOS** codebase (Vectorized Architecture for Multiobjective Optimization Studies).

This file explains how to set up the environment, how the project is structured, and what conventions to follow when changing code. README.md is the source of truth for capabilities and commands; this document translates that into expectations for contributors and AI agents.

---

## 1. What VAMOS is

- Research-grade, vectorized multi-objective optimization framework in Python (Python 3.10+).
- Algorithms: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO with vectorized kernels (NumPy, Numba, MooCore).
- Encodings: continuous, permutation, binary, integer, mixed; operators are vectorized and workspace-aware.
- Problems: ZDT, DTLZ, WFG, LZ09, CEC2009 UF/CF (pymoo-backed), TSP/TSPLIB, feature selection/knapsack/QUBO, allocation/job assignment, welded beam, ML/real-data examples.
- Tooling and orchestration: CLI (`python -m vamos.experiment.cli.main`), StudyRunner (problem x algorithm x seed sweeps), config-driven runs (YAML/JSON), diagnostics (`vamos-self-check`), benchmarking (`vamos-benchmark`), Studio (`vamos-studio`) for interactive decision-making, visualization/MCDM/stats utilities, and hypervolume early-stop options.
- Output lives under `results/` by default (fronts, metadata).

---

## 2. Environment and install

- Create a virtual environment:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- Typical editable install (core + backends + benchmarks + dev/test):

  ```powershell
  pip install -e ".[backends,benchmarks,dev]"
  ```

- Extras you may need: `backends`, `benchmarks`, `dev`, `notebooks`, `examples`, `docs`, `studio`, `analytics`, `autodiff` (see README for commands such as `pip install -e ".[backends,benchmarks,dev,notebooks]"`).

- Useful commands:
  - Quick run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
  - MooCore backend: `python -m vamos.experiment.cli.main --engine moocore --problem zdt1 --max-evaluations 2000`
  - Benchmark suite: `vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/`
  - Diagnostics: `python -m vamos.experiment.diagnostics.self_check` or `vamos-self-check`
  - Tests: `pytest` (full optional stack: `pytest -m "not slow"` after installing the relevant extras)
  - Numba variation toggle: `VAMOS_USE_NUMBA_VARIATION=1`
  - Library imports: prefer the root facades (`vamos.api`, `vamos.algorithms`, `vamos.problems`, `vamos.plotting`, `vamos.mcdm`, `vamos.stats`, `vamos.tuning`); contributor work can use layered packages directly.

---

## 3. Architecture & layers

- `foundation`: base abstractions, constraints, eval/metrics, kernels/backends, problem definitions/registries, packaged data, and version helpers.
- `engine`: evolutionary/search machinery — algorithms, operators, hyperheuristics, tuning pipelines, and algorithm configs.
- `experiment`: orchestration and entry points — CLI, StudyRunner, benchmark suites, diagnostics, and zoo presets.
- `ux`: user-facing analysis/visualization (including MCDM/stats) and Studio surfaces.
- The `vamos` package root re-exports a curated public API (see `vamos.api`, `vamos.algorithms`, `vamos.problems`, `vamos.plotting`, `vamos.mcdm`, `vamos.stats`, `vamos.tuning`); use these for user-facing examples.
- Standard results layout under `results/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/` with `FUN.csv`, optional `X.csv`/`G.csv`/archive files, `metadata.json`, and `resolved_config.json`. All experiment runners (CLI/benchmark/zoo) must write to this schema; UX helpers (`vamos.ux.analysis.results`) read it.
- Dependency rules:
  - `ux` may depend on `experiment`, `engine`, `foundation`.
  - `experiment` may depend on `engine`, `foundation`.
  - `engine` may depend on `foundation`.
  - `foundation` should stay free of upward dependencies; keep cross-layer facades minimal and documented.
- Place new code in the correct layer and keep imports aligned with these directions.
- Only re-export stable, well-documented constructs from the root package; extension work should live in the layered modules.

### Tests and markers
- Tests are organized by layer: `tests/foundation`, `tests/engine`, `tests/experiment`, `tests/ux`, with `tests/integration` for cross-layer scenarios.
- Markers: `smoke` (fast critical), `slow` (long), `backends` (optional backends), `notebooks`, `examples`, plus `cli`/`numba`/`moocore`/`studio`/`autodiff` where applicable.
- Default quick run: `pytest -m "smoke"`. Tag backend/notebook-heavy tests appropriately to keep smoke runs fast.

---

## 4. Repository layout (aligns with README)

- `src/vamos/`
  - `foundation/` - core orchestration utilities, constraints, eval/metrics, kernel backends, problems/registries, shared data, version helpers.
  - `engine/` - algorithms, operators, hyperheuristics, tuning stacks, variation/config helpers.
  - `experiment/` - CLI entrypoints, study runner, benchmark/reporting, diagnostics, zoo presets.
  - `ux/` - analysis/MCDM/stats helpers, visualization, Studio.
- `tests/` - pytest suite (operators, algorithms, CLI/study integration, examples/notebooks when enabled).
- `examples/`, `notebooks/` - runnable examples and exploratory notebooks.
- Results and reports default to `results/` (fronts, metadata, CSVs).

Prefer following this structure instead of guessing new locations.

---

## 5. Coding conventions

- Match existing style per module; add type hints and explicit imports.
- Keep vectorization and workspace usage; avoid Python loops in hot paths.
- Preserve public APIs; if a breaking change is unavoidable, update tests/docs and check all call sites.
- Seed stochastic components for reproducibility in tests and examples.
- Add concise comments only when intent is non-obvious (avoid noise).

---

## 6. Dependency and performance contract

- Do NOT add new heavy dependencies (large ML frameworks, extra plotting stacks, etc.) unless strictly necessary and clearly justified.
- Prefer existing extras (`backends`, `benchmarks`, `notebooks`, `examples`, `studio`, etc.) instead of inventing new ones.
- Preserve vectorization and the existing kernels (NumPy/Numba/MooCore); avoid replacing them with pure Python loops in critical paths.
- Any new dependency must:
  - Be optional if not core to the framework.
  - Be wired through `pyproject.toml` extras (or equivalent).
  - Be exercised by at least one test or example.

---

## 7. Style and linting

The preferred toolchain is:

- `black` for formatting
- `ruff` for linting
- `isort` for import ordering
- `pytest` for tests

When adding or modifying code, aim for code that passes:

```bash
black src tests
ruff check src tests
isort src tests
pytest
```

Use `pre-commit` hooks if configured in the repo.

---

## 8. Testing and QA

- Run `pytest` after changes; for broad coverage with extras, use `pytest -m "not slow"` (per README guidance).
- If you touch core algorithms/operators/kernels, add or update smoke tests (small budgets) and invariant checks.
- For new outputs/metadata, ensure schemas remain additive and compatible.

---

## 9. Workflow for AI coding agents

1. Load context: read this file, `README.md`, `AGENTS_tasks.md`, and any relevant docs/examples.
2. Plan minimal, focused changes; avoid wide refactors unless requested.
3. Respect dependency boundaries and performance constraints.
4. Document non-obvious decisions; keep defaults stable unless justified.
5. Ensure tests relevant to your change pass or explain why they were not run.

---

## 10. Common tasks

- See `AGENTS_tasks.md` for task-specific playbooks (operators, problems, tuning, studies, benchmarking, diagnostics).

---

## 11. Non-goals and artifacts

- Do not rewrite the architecture or remove vectorization/workspaces in critical paths.
- Do not commit large generated artifacts (plots, `.npz`, etc.) under `src/`; keep outputs under `results/`/`report/`/`target/`.

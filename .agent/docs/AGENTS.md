# AGENTS.md

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


> For a user-facing overview of VAMOS (features, install, CLI examples), see `README.md`.
> For contributor workflows and concrete tasks, see `AGENTS_tasks.md`.
> For ready-to-paste prompts for code assistants, see `AGENTS_codex_prompts.md`.

Guidance for AI coding agents (Codex, GPT, Copilot, etc.) working on the **VAMOS** codebase (Vectorized Architecture for Multiobjective Optimization Studies).

This file explains how to set up the environment, how the project is structured, and what conventions to follow when changing code. `../../README.md` is the source of truth for capabilities and commands; this document translates that into expectations for contributors and AI agents.

---

## 1. What VAMOS is

- Research-grade, vectorized multi-objective optimization framework in Python (Python 3.10+).
- **Unified API**: All workflows should use `vamos.optimize()` as the primary entry point.
- **Algorithms**: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA with vectorized kernels.
- **Tooling**: Comprehensive suite via `vamos <subcommand>`: `vamos profile` (profiling), `vamos bench` (reporting), `vamos studio` (interactive dashboard).
- **Notebooks**: extensive examples under `notebooks/0_basic/`, `notebooks/1_intermediate/` and `notebooks/2_advanced/`.

### User-Friendliness Principles

VAMOS prioritizes ease of use:

1.  **Unified Entry Point**: Prefer `vamos.optimize(...)` for user-facing examples; use algorithm config objects for full control.
    ```python
    from vamos import optimize

    result = optimize("zdt1", algorithm="nsgaii", budget=2000, pop_size=100, seed=42)
    ```
    ```python
    from vamos import optimize
    from vamos.algorithms import NSGAIIConfig
    from vamos.problems import ZDT1

    problem = ZDT1(n_var=30)
    cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 2000),
        seed=42,
    )
    ```

2.  **Friendly Custom Problems**: Use `make_problem()` for custom problems -- no class boilerplate, no NumPy vectorization knowledge required. VAMOS auto-vectorizes scalar functions internally.
    ```python
    from vamos import make_problem, optimize

    problem = make_problem(
        lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
        n_var=2, n_obj=2,
        bounds=[(0, 1), (0, 1)],
    )
    result = optimize(problem, algorithm="nsgaii", max_evaluations=5000)
    ```
    For scaffolding a problem file interactively, use `vamos create-problem`.
    For a visual approach, use the **Problem Builder** tab in VAMOS Studio (`vamos studio`).

3.  **Interactive Results**: Use `explore_result_front(result)` for immediate visualization in notebooks.
4.  **Publication Ready**: Use `result_to_latex(result)` for generating tables directly from code.
5.  **No Internal Imports**: Avoid deep internal modules (`vamos.engine.algorithm...`, `vamos.foundation...`). Use the public facades: `vamos`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`.

---

## 2. Environment and install

- **Standard Install**:
  ```powershell
  pip install -e ".[backends,benchmarks,dev,notebooks]"
  ```

- **Useful commands**:
  - Quick run: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000`
  - Self-check: `vamos check`
  - Profile: `vamos profile nsgaii zdt1`
  - Tune: `vamos tune --problem zdt1 --algorithm nsgaii --budget 5000 --tune-budget 200 --n-seeds 5`
    (`--tune-budget` counts configuration evaluations; `--budget` is per-run evaluations.)
  - Tests: `pytest`

- **Library imports (user-facing)**:
  ```python
  from vamos import optimize, make_problem, make_problem_selection
  from vamos.algorithms import NSGAIIConfig
  from vamos.ux.api import plot_pareto_front_2d, weighted_sum_scores
  ```
- **Library imports (contributors)**: Layered packages for internal work:
  ```python
  from vamos.engine.algorithm.nsgaii import NSGAII
  from vamos.foundation.problem.registry import PROBLEM_SPECS
  ```

---

## 3. Architecture & layers

- `foundation`: base abstractions, constraints, eval/metrics, kernels/backends, problem definitions/registries, packaged data, and version helpers.
- `engine`: evolutionary/search machinery — algorithms, operators, hyperheuristics, tuning pipelines, and algorithm configs.
- `experiment`: orchestration and entry points — CLI, StudyRunner, benchmark suites, diagnostics, and zoo presets.
- `ux`: user-facing analysis/visualization (including MCDM/stats) and Studio surfaces.
- The `vamos` package root re-exports a curated public API (see `vamos.api`, `vamos.algorithms`, `vamos.problems`, and `vamos.ux.api`); use these for user-facing examples.
- Standard results layout under `results/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/` with `FUN.csv`, optional `X.csv`/`G.csv`/archive files, `metadata.json`, and `resolved_config.json`. All experiment runners (CLI/benchmark/zoo) must write to this schema; UX helpers (`vamos.ux.analysis.results`) read it.
---

## 4. Repository layout (aligns with README)

- `src/vamos/`
  - `foundation/` - core abstractions, constraints, eval/metrics, kernel backends, problems/registries, shared data, version helpers. Includes `exceptions.py`.
  - `engine/` - algorithms, operators, hyperheuristics, tuning stacks, variation/config helpers.
  - `experiment/` - CLI entrypoints, study runner, benchmark/reporting, diagnostics, zoo presets. Includes `experiment_context.py` and `quick.py`.
  - `ux/` - analysis/MCDM/stats helpers, visualization, Studio.
  - `api.py` - Root facade. Other facades (`algorithms`, `problems`, `ux.api`) are public.
- `tests/` - pytest suite (operators, algorithms, CLI/study integration, examples/notebooks when enabled).
- `examples/`, `notebooks/` - runnable examples and exploratory notebooks.
- `notebooks/90_paper_benchmarking.ipynb` includes SAES-style critical distance plots (toggle with `CD_STYLE`).
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

1. Load context: read this file, `../../README.md`, `AGENTS_tasks.md`, and any relevant docs/examples.
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

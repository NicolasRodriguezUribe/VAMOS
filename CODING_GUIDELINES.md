# CODING_GUIDELINES

These notes capture the conventions we rely on when extending VAMOS (Vectorized Architecture for Multiobjective Optimization Studies). Please follow them for any code, notebooks, or documentation that will live in the repository.

## User-Friendliness Principles

VAMOS is designed to be **user-friendly first**. Every design decision should consider the end-user experience:

1. **Clean public API**: Export stable symbols from `vamos` root. Users should write:
   ```python
   from vamos import optimize
   from vamos.algorithms import NSGAIIConfig
   from vamos.problems import ZDT1
   from vamos.ux.api import plot_pareto_front_2d
   ```
   Not internal paths like `from vamos.engine.algorithm.config import NSGAIIConfig`.

2. **Sensible defaults**: New features should work out-of-the-box. Require minimal configuration for basic use cases.

3. **Consistent patterns**: Follow existing API conventions (builder pattern for configs, `optimize()` for running, etc.).

4. **Clear errors**: Raise helpful exceptions with suggestions, not cryptic tracebacks.

5. **Documentation**: Every public API needs docstrings and examples.

### When Adding New Features

- Export user-facing symbols in `vamos/__init__.py` and domain modules (`vamos/problems.py`, etc.)
- Keep internal implementation in layered packages (`foundation`, `engine`, `experiment`, `ux`)
- Write examples using only the public API to verify usability
- Test that imports from `vamos` work before merging

## Architecture & layers
- `foundation`: base abstractions, constraints, eval/metrics utilities, kernel backends, problems/registries, packaged data, and version helpers.
- `engine`: search machinery — algorithms, operators, hyperheuristics, tuning pipelines, and algorithm configs.
- `experiment`: orchestration — CLI entrypoints, StudyRunner, benchmark/reporting code, diagnostics, and zoo presets.
- `ux`: user-facing analysis/analytics, visualization, Studio, and MCDM/stats helpers.
- Dependency rules: `ux` -> `experiment`/`engine`/`foundation`; `experiment` -> `engine`/`foundation`; `engine` -> `foundation`; keep `foundation` free of upward dependencies and place cross-layer facades intentionally.

## Development workflow
- Target Python **3.10+** (match `pyproject.toml`) and keep contributions inside `src/vamos` to preserve the single import root.
- Use feature branches, keep commits logically scoped, and prefer descriptive commit messages (`algo: clarify NSGA-II init`). Commit titles and descriptions must be written in English and briefly summarize the change so pushes remain self-documenting.
- Run the local toolchain before opening a PR:
  - `ruff check src tests` for linting/quality gates.
  - `black src tests` for formatting (line length 88, default profile).
  - `pytest` for smoke/benchmark suites. Add new tests when behaviour changes or regressions are possible.
- Keep notebooks in `notebooks/` and do not rely on them for reproducible logic. If code becomes reusable, promote it into `src/vamos` and add tests.

## Python style
- Follow PEP 8/PEP 257 with **type hints everywhere**. Prefer explicit return types on public functions/classes.
- Use modern typing (PEP 585/604): `list`/`dict`/`tuple`/`set` generics and `| None`/`|` unions; avoid `typing.List/Dict/Optional/Union` in new code.
- Structure imports in three blocks (stdlib, third-party, local) and avoid wildcard imports.
- Write pure/side-effect-free helpers when practical; keep state inside dataclasses or small objects (see `NSGAII`).
- When dealing with randomness, use `numpy.random.Generator` instances passed through call stacks (`np.random.default_rng(seed)`) instead of global RNGs.
- Raise `ValueError`/`TypeError` for invalid configuration instead of silent fallbacks. Validate dictionary-based configs early (e.g., `_prepare_mutation_params`).
- Prefer vectorized NumPy kernels or dedicated backend hooks over Python loops. If Numba/MooCore features are required, gate them via the backend registry and document any limitations.
- Keep new public APIs fully typed; prefer Protocols for shared contracts (Problem, KernelBackend, Algorithm). Run `mypy src/vamos/foundation/core src/vamos/engine/algorithm src/vamos/foundation/kernel` when touching those areas.

## Project structure hints
- `src/vamos/foundation/` hosts core primitives, constraints, eval/metrics, kernels, problem definitions/registries, packaged data, and version helpers.
- `src/vamos/engine/` hosts algorithms, operators, hyperheuristics, tuning, and algorithm config utilities.
- `src/vamos/experiment/` hosts CLI/study runners, benchmark/reporting utilities, diagnostics, and the experiment zoo.
- `src/vamos/ux/` hosts analysis/MCDM/stats utilities, visualization helpers, and Studio.
- Keep module-level docstrings describing the problem, algorithm, or kernel where appropriate.
- Use `docs/` for architecture/design notes, `results/` for generated experiment artefacts, and `tests/` for pytest modules mirroring the `src` layout.
- For configuration-style modules (e.g., problem registries), expose typed dictionaries or dataclasses so downstream scripts can rely on structured data.

## Documentation and comments
- Keep README high level. Use `CODING_GUIDELINES.md` (this file) for developer-specific rules and `docs/` for deeper design discussions.
- Update `docs/` when adding or changing user-facing behaviour (CLI flags, configs, algorithms, problems). Use `mkdocs serve` to preview.
- Write short, purposeful comments only where intent is not obvious. Prefer descriptive names over verbose comments.
- When adding equations or algorithm details, include references (paper name, section) to help future contributors.
- All public-facing content (code, comments, docstrings, commit messages, notebooks) must be written in English; avoid multilingual snippets that could confuse future maintainers.

## Future features & components
- **Always write implementation code, tests, and documentation in English.** If a legacy snippet is in another language, translate it while touching the file.
- **Research before building.** Prior to adding any operator, problem, kernel, or study tooling, review the reference implementations in jMetal, jMetalPy, and pymoo. If an equivalent feature exists, ensure our version covers the same capabilities (or document why it intentionally differs).
- **Favor parity, then optimization.** First match the feature set and behaviour you observed in the reference libraries, including configuration flags and sensible defaults. Once the functionality is correct, profile the new code path (e.g., `pytest -k feature --durations=20`, `python -m cProfile`) and optimize hot spots via vectorization, backend hooks, or caching.
- **Record the research outcome.** Capture short notes in the PR description or module docstring summarizing which library reference you followed and any deliberate deviations, so future contributors can trace the design decision quickly.
- **Keep extensibility in mind.** Structure new components so future variants (e.g., alternative crossover strategies) can plug in via the existing registries without large refactors.

## Testing & benchmarking
- Unit tests go under `tests/` and should focus on deterministic components (operator builders, kernel utilities, registries).
- For stochastic algorithms, add property-based or statistical checks (distribution shape, constraint violations) rather than brittle exact matches.
- Reproduce any new benchmark/regression input data via scripts stored under `data/` or `notebooks/`. If results are stored in `results/`, ensure metadata describes the configuration so runs are comparable.

### Test layout and markers
- Tests mirror the layered architecture: `tests/foundation`, `tests/engine`, `tests/experiment`, `tests/ux`, with `tests/integration` for cross-layer coverage.
- Markers: `smoke` (fast, critical), `slow` (long runs), `backends` (optional backends like numba/moocore), `notebooks` (notebook-dependent), `examples` (example scripts), `cli`, `numba`, `moocore`, `studio`, `autodiff`.
- Quick check: `pytest -m "smoke"`; full: `pytest` or targeted markers as needed.
- Place new tests in the appropriate layer and tag backend- or notebook-dependent cases to keep default runs fast.

## Tooling checklist (pre-PR)
1. `pip install -e .[dev]`
2. `ruff check src tests`
3. `black src tests`
4. `pytest`
5. Verify README/docs updates when the public interface changes.

Following these conventions keeps the project fast to iterate on, reproducible, and pleasant for future contributors. Happy optimizing!

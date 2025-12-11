# CODING_GUIDELINES

These notes capture the conventions we rely on when extending VAMOS (Vectorized Architecture for Multiobjective Optimization Studies). Please follow them for any code, notebooks, or documentation that will live in the repository.

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
- Structure imports in three blocks (stdlib, third-party, local) and avoid wildcard imports.
- Write pure/side-effect-free helpers when practical; keep state inside dataclasses or small objects (see `NSGAII`).
- When dealing with randomness, use `numpy.random.Generator` instances passed through call stacks (`np.random.default_rng(seed)`) instead of global RNGs.
- Raise `ValueError`/`TypeError` for invalid configuration instead of silent fallbacks. Validate dictionary-based configs early (e.g., `_prepare_mutation_params`).
- Prefer vectorized NumPy kernels or dedicated backend hooks over Python loops. If Numba/MooCore features are required, gate them via the backend registry and document any limitations.
- Keep new public APIs fully typed; prefer Protocols for shared contracts (Problem, KernelBackend, Algorithm). Run `mypy src/vamos/core src/vamos/algorithm src/vamos/kernel` when touching those areas.

## Project structure hints
- Algorithms live in `src/vamos/algorithm/`; kernels go under `src/vamos/kernel/`; benchmark definitions stay in `src/vamos/problem/`. Keep module-level docstrings describing the problem, algorithm, or kernel.
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

## Tooling checklist (pre-PR)
1. `pip install -e .[dev]`
2. `ruff check src tests`
3. `black src tests`
4. `pytest`
5. Verify README/docs updates when the public interface changes.

Following these conventions keeps the project fast to iterate on, reproducible, and pleasant for future contributors. Happy optimizing!

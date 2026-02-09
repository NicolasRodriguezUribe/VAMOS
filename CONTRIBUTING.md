# Contributing to VAMOS

Thank you for considering a contribution! This project is organized to make adding new components straightforward.

## Adding a new problem
- Put the implementation under `src/vamos/foundation/problem/` (one file per problem family).
- Register it by adding a spec to the correct family module under `src/vamos/foundation/problem/registry/families/`.
- Add a short docstring to the problem class describing the landscape and encoding.
- Add a small smoke test in `tests/` to validate instantiation and `evaluate` shape.
- See `docs/dev/add_problem.md` for a step-by-step template.

## Adding a new algorithm
- Implement the vectorized core under `src/vamos/engine/algorithm/`.
- Create a config dataclass/builder under `src/vamos/engine/algorithm/config/` to keep construction declarative and serializable.
- Register the algorithm in `src/vamos/engine/algorithm/registry.py` so orchestration layers and the CLI can resolve it by name.
- Add a minimal smoke test (tiny population/evaluation budget) to catch wiring issues.
- See `docs/dev/add_algorithm.md` for a template and checklist.

## Adding a new kernel backend
- Implement the `KernelBackend` interface in `src/vamos/foundation/kernel/` (see `kernel/backend.py` for required methods).
- Register it in `src/vamos/foundation/kernel/registry.py` with a unique engine name.
- Add a backend-marked smoke test (`@pytest.mark.<engine>`) that runs a small NSGA-II job; use `pytest.importorskip` to skip when the dependency is missing.
- See `docs/dev/add_backend.md` for required methods and a smoke-test example.

## Architecture health (mandatory)
- Read the ADRs before any architectural change: `docs/dev/adr/index.md`.
- Run the local health command (same gates as CI): `python tools/health.py`.
- Optional strict typing: `python tools/health.py --mypy-full` (stricter than CI).
- If you change public APIs, update the snapshot: `python tools/update_public_api_snapshot.py`.

## Continuous Integration
- CI runs the architecture health gates, ruff lint/format, mypy budget, build smoke, and full tests.
- Before opening a PR, run the same locally:
  - `python tools/health.py`
  - `pytest -q`

## Coding style and typing
- The project uses a `src/` layout and prefers type hints on public-facing functions/classes.
- Performance-critical loops (kernels, variation) should remain lightweight; avoid refactors that change behavior without explicit benchmarks.
- New orchestration/config/registry code should be mypy-friendly; see `pyproject.toml` for incremental typing settings.

## Tuning package layout
- All tuning utilities (parameter spaces, samplers, racing loop, random search) live under `src/vamos/engine/tuning/racing/`.
- Import from `vamos.engine.tuning` or use the user-friendly facade `vamos.tuning`.

## Self-check
- After changes, run `vamos check` for a quick sanity check.
- CI-friendly tests live under `tests/`; keep populations/evaluation budgets small for speed.

## Before opening a pull request (human or AI-assisted)
1. Run the health gates: `python tools/health.py`.
2. Run the full suite: `pytest -q`.
3. If you touched tuning, StudyRunner, or benchmarking:
   - Run the smallest relevant `vamos bench` suite.
4. If you added docs or notebooks:
   - Build docs locally (`mkdocs serve`) or open the notebook and run all cells.
5. For guidance on assistant-specific workflows, see `.agent/docs/AGENTS.md`, `.agent/docs/AGENTS_tasks.md`, and `README.md`.

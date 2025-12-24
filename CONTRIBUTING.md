# Contributing to VAMOS

Thank you for considering a contribution! This project is organized to make adding new components straightforward.

## Adding a new problem
- Put the implementation under `src/vamos/foundation/problem/` (one file per problem family).
- Expose the class via the registry in `src/vamos/foundation/problems_registry.py` by adding a `ProblemSpec` entry (set sensible defaults for `n_var`, `n_obj`, `encoding`, and whether `n_obj` is overridable).
- Add a short docstring to the problem class describing the landscape and encoding.
- Add a small smoke test in `tests/` to validate instantiation and `evaluate` shape.
- See `docs/dev/add_problem.md` for a step-by-step template.

## Adding a new algorithm
- Implement the vectorized core under `src/vamos/engine/algorithm/`.
- Create a config dataclass/builder in `src/vamos/engine/algorithm/config.py` to keep construction declarative and serializable.
- Register the algorithm in `src/vamos/engine/algorithms_registry.py` so orchestration layers and the CLI can resolve it by name.
- Add a minimal smoke test (tiny population/evaluation budget) to catch wiring issues.
- See `docs/dev/add_algorithm.md` for a template and checklist.

## Adding a new kernel backend
- Implement the `KernelBackend` interface in `src/vamos/foundation/kernel/` (see `kernel/backend.py` for required methods).
- Register it in `src/vamos/foundation/kernel/registry.py` with a unique engine name.
- Add a backend-marked smoke test (`@pytest.mark.<engine>`) that runs a small NSGA-II job; use `pytest.importorskip` to skip when the dependency is missing.
- See `docs/dev/add_backend.md` for required methods and a smoke-test example.

## Continuous Integration
- CI (GitHub Actions) runs `ruff`, `black --check`, and `pytest -m "not slow"` on Python 3.10–3.12 with full extras installed.
- A minimal-install job installs the core package only and runs smoke tests to ensure optional dependencies are lazy.
- A wheel build job checks that packaged data (reference fronts, weights) are present.
- Before opening a PR, run the same locally:
  - `pip install -e ".[dev,backends,benchmarks,notebooks,studio,autodiff]"`
  - `ruff check .`
  - `black --check .`
  - `pytest -m "not slow"`
- Mypy: type hints are required on new public APIs. Run `mypy src/vamos/foundation/core src/vamos/engine/algorithm src/vamos/foundation/kernel` when touching those areas.

## Coding style and typing
- The project uses a `src/` layout and prefers type hints on public-facing functions/classes.
- Performance-critical loops (kernels, variation) should remain lightweight; avoid refactors that change behavior without explicit benchmarks.
- New orchestration/config/registry code should be mypy-friendly; see `pyproject.toml` for incremental typing settings.

## Tuning package layout
- All tuning utilities (parameter spaces, samplers, racing loop, random search) live under `src/vamos/engine/tuning/racing/`.
- Import from `vamos.engine.tuning` or use the user-friendly facade `vamos.tuning`.

## Self-check
- After changes, run `python -m vamos.experiment.diagnostics.self_check` or `vamos-self-check` for a quick sanity check.
- CI-friendly tests live under `tests/`; keep populations/evaluation budgets small for speed.

### Before opening a pull request (human or AI-assisted)

1. Ensure you can run `pytest` locally (or at least the relevant subset).
2. Run a small CLI smoke test, for example:
   ```powershell
   python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 1000
   ```
3. If you touched tuning, StudyRunner, or benchmarking:
   - Run the smallest relevant `vamos-benchmark` suite.
4. If you added docs or notebooks:
   - Build docs locally (`mkdocs serve`) or open the notebook and run all cells.
5. For guidance on assistant-specific workflows, see `.agent/docs/AGENTS.md`, `.agent/docs/AGENTS_tasks.md`, and `README.md`.

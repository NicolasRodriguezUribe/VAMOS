# Contributing to VAMOS

Thank you for considering a contribution! This project is organized to make adding new components straightforward.

## Adding a new problem
- Put the implementation under `src/vamos/problem/` (one file per problem family).
- Expose the class via the registry in `src/vamos/problem/registry.py` by adding a `ProblemSpec` entry (set sensible defaults for `n_var`, `n_obj`, `encoding`, and whether `n_obj` is overridable).
- Add a short docstring to the problem class describing the landscape and encoding.
- Add a small smoke test in `tests/` to validate instantiation and `evaluate` shape.

## Adding a new algorithm
- Implement the vectorized core under `src/vamos/algorithm/`.
- Create a config dataclass/builder in `src/vamos/algorithm/config.py` to keep construction declarative and serializable.
- Register the algorithm in `src/vamos/algorithm/registry.py` so orchestration layers and the CLI can resolve it by name.
- Add a minimal smoke test (tiny population/evaluation budget) to catch wiring issues.

## Adding a new kernel backend
- Implement the `KernelBackend` interface in `src/vamos/kernel/` (see `kernel/backend.py` for required methods).
- Register it in `src/vamos/kernel/registry.py` with a unique engine name.
- Add a backend-marked smoke test (`@pytest.mark.<engine>`) that runs a small NSGA-II job; use `pytest.importorskip` to skip when the dependency is missing.

## Coding style and typing
- The project uses a `src/` layout and prefers type hints on public-facing functions/classes.
- Performance-critical loops (kernels, variation) should remain lightweight; avoid refactors that change behavior without explicit benchmarks.
- New orchestration/config/registry code should be mypy-friendly; see `pyproject.toml` for incremental typing settings.

## Self-check
- After changes, run `python -m vamos.self_check` or `vamos-self-check` for a quick sanity check.
- CI-friendly tests live under `tests/`; keep populations/evaluation budgets small for speed.

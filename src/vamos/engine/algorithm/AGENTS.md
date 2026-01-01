# Algorithm package (engine layer)

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).


This directory contains VAMOS's algorithm implementations plus shared algorithm
building blocks used across multiple algorithms.

## Structure

- `components/`: reusable components
  - `archive.py`: external archives (crowding distance, hypervolume)
  - `population.py`: population initialization and evaluation helpers
  - `selection.py`: parent selection strategies
  - `termination.py`: termination criteria / trackers (e.g. `HVTracker`)
  - `hypervolume.py`: hypervolume utilities (with backend fallbacks)
  - `weight_vectors.py`: weight vectors (NSGA-III / MOEA-D)
  - `variation/`: variation pipelines (crossover + mutation wiring)
- Algorithm subfolders: `nsgaii/`, `moead/`, `spea2/`, `ibea/`, `smsemoa/`, `smpso/`, `nsgaiii/`
  - Each contains: `__init__.py`, `{algorithm}.py`, `initialization.py`, `helpers.py`, `operators.py`, `state.py`
- Config subfolder: `config/`
  - `base.py`, `nsgaii.py`, `moead.py`, `spea2.py`, `ibea.py`, `smsemoa.py`, `smpso.py`, `nsgaiii.py`
- Registry/factory: `registry.py`, `factory.py`, `builders.py`

## Conventions

- Prefer an ask/tell-style loop and keep all hot paths vectorized (NumPy/Numba).
- Populations are arrays, not per-individual objects:
  - `X`: `(pop_size, n_var)` decision variables
  - `F`: `(pop_size, n_obj)` objective values
  - `G`: `(pop_size, n_constraints)` constraint violations (optional)

## Adding a new algorithm

1. Implement `my_algo.py` (or a subpackage if it grows).
2. Add config dataclass + builder in `config.py`.
3. Register in `registry.py` with a stable lowercase algorithm id.
4. Add tests under `tests/engine/` (mark fast ones with `@pytest.mark.smoke`).

## Notes

- Do not add compatibility shims at this level; reuse `components/*` instead.
- Delegate expensive operations to kernels (`problem.evaluate`, `kernel.*`) or to
  shared utilities in `components/`.

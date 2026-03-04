# Algorithm package (engine layer)

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


This directory contains VAMOS's algorithm implementations plus shared algorithm
building blocks used across multiple algorithms.

## Structure

- `components/`: reusable components
  - `archive.py`: external archives and bounded-archive pruning policies (`crowding`, `hv`, `mc_hv`, `knn`, `maxmin`, `ref_dirs`)
  - `population.py`: population initialization and evaluation helpers
  - `selection.py`: parent selection strategies
  - `termination.py`: termination criteria / trackers (e.g. `HVTracker`)
  - `hypervolume.py`: hypervolume utilities (with backend fallbacks)
  - `weight_vectors.py`: weight vectors (NSGA-III / MOEA-D)
  - `variation/`: variation pipelines (crossover + mutation wiring)
- Algorithm subfolders: `nsgaii/`, `moead/`, `spea2/`, `ibea/`, `smsemoa/`, `smpso/`, `nsgaiii/`, `agemoea/`, `rvea/`
  - Each contains: `__init__.py`, `{algorithm}.py`, `initialization.py`, `helpers.py`, `state.py`
  - Operator wiring lives in `src/vamos/engine/operators/policies/`
- Config subfolder: `config/`
  - `base.py`, `nsgaii.py`, `moead.py`, `spea2.py`, `ibea.py`, `smsemoa.py`, `smpso.py`, `nsgaiii.py`, `agemoea.py`, `rvea.py`
  - **Unified Archive API**: All configs must support `.external_archive(capacity, **kwargs)` using `ExternalArchiveConfig`.
  - Archive-enabled configs default to archive-backed top-level results unless `result_mode("population")` is set explicitly.
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

## Tuning Integration

Algorithms are tunable via `vamos.engine.tuning.racing`:
- Use `build_{algo}_config_space()` from `bridge.py` to get parameter space
- Use `config_from_assignment(algo_name, params)` to convert tuned params to config
- For tuned bounded external archives, the capacity is the population size; there is no `archive_size_factor` in the public tuning space.
- Multi-fidelity tuning passes varying `budget` via `EvalContext`
- Warm-start support: algorithms can checkpoint population state for continuation



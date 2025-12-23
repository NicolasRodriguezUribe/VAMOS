# AGENTS_tasks.md

Task playbook for AI coding agents working on **VAMOS**

This document complements `AGENTS.md`. It defines concrete, common tasks and what an AI agent should do (and avoid) for each one. This playbook assumes you have already read `README.md` and `AGENTS.md`. It focuses on concrete, well-scoped tasks that can be safely delegated to humans or AI coding agents.

## User-Friendliness First

VAMOS is designed to be **user-friendly**. When writing code, examples, or documentation:

- **Always use the public API** for user-facing code: `from vamos import ...`
- **Avoid exposing internal paths** to end users (reserve `vamos.engine.*`, `vamos.foundation.*` for contributors)
- **Prefer sensible defaults** so basic usage requires minimal configuration
- **Write clear error messages** that guide users to solutions

### Import Pattern Summary

```python
# USER-FACING CODE (examples, notebooks, scripts)
from vamos import (
    optimize, OptimizeConfig, NSGAIIConfig,
    ZDT1, make_problem_selection,
    plot_pareto_front_2d, weighted_sum_scores,
)

# CONTRIBUTOR CODE (internal modules, tests)
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.registry import PROBLEM_SPECS
```

User-facing imports should go through the root facades (`vamos.api`, `vamos.algorithms`, `vamos.problems`, `vamos.plotting`, `vamos.mcdm`, `vamos.stats`, `vamos.tuning`). Contributor work can target the layered packages (`foundation`, `engine`, `experiment`, `ux`) directly. All experiment outputs should follow the standard layout: `<output_root>/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/` with `FUN.csv`, optional `X.csv`/`G.csv`/archive files, `metadata.json`, and `resolved_config.json`.
When working on the paper benchmarking notebook (`notebooks/11_paper_benchmarking.ipynb`), keep the SAES-style critical distance plot toggle (`CD_STYLE`) intact.
Tests mirror the layers: `tests/foundation`, `tests/engine`, `tests/experiment`, `tests/ux`, and `tests/integration` for cross-layer checks. Markers: `smoke`, `slow`, `backends`, `notebooks`, `examples`, `cli`, `numba`, `moocore`, `studio`, `autodiff`. Use `pytest -m "smoke"` for quick verification.

### Task index

| Task ID | What it does                                 | Useful commands / entry points                      |
|--------:|---------------------------------------------|-----------------------------------------------------|
| T1      | Baseline run on ZDT1                        | `python -m vamos.experiment.cli.main --problem zdt1 ...`       |
| T2      | Add/modify external archive                 | `python -m vamos.experiment.cli.main --hv-threshold ...`       |
| T3      | Extend AutoNSGA-II / config space           | `python -m vamos.experiment.cli.main --config path.yaml`       |
| T4      | New real-coded operator                     | `python -m vamos.experiment.cli.main` on continuous problems   |
| T5      | New benchmark problem                       | `python -m vamos.experiment.cli.main --problem <id> ...`       |
| T6      | Create study / experiment pipeline          | StudyRunner/central runner / `python -m vamos.experiment.cli.main --config`   |
| T7      | Improve logging/metadata + analysis helpers | Outputs under standard results layout; UX loaders    |
| T8      | Add to benchmarking CLI                     | `vamos-benchmark --suite ... --output ...`          |
| T9      | Extend diagnostics / self-check             | `python -m vamos.experiment.diagnostics.self_check`            |

---

## 1. Run a baseline experiment on ZDT1

**Goal**: Verify installation and core algorithms end-to-end using a tiny NSGA-II run on ZDT1.

**Context**: `README.md` quickstart; `src/vamos/foundation/problem/` for ZDT problems; `src/vamos/engine/algorithm/` for NSGA-II; CLI under `vamos.experiment.cli.main`.

**Steps**
- Use the CLI: `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations 2000` (or lower for CI).
- Optionally switch engine/backends: `--engine moocore` after installing `[backends]`.
- If adding a smoke test, keep budgets tiny (few generations/population) and assert non-empty finite fronts.

**Avoid**: New dependencies or hardcoded paths.

---

## 2. Add or modify an external archive (e.g., crowding vs. hypervolume)

**Goal**: Implement or adjust an external archive (crowding/hypervolume) and integrate it without breaking defaults.

**Context**: Archive utilities in `src/vamos/foundation/eval/`, `src/vamos/foundation/metrics/`, or `src/vamos/engine/algorithm/` hooks; hypervolume helpers in kernels/backends; algorithm configs that accept archive choices.

**Steps**
- Locate archive abstraction and existing implementations (capacity, `add`/`extend` behaviour).
- Implement the new archive (e.g., hypervolume contribution) with graceful fallback if hypervolume backends are missing.
- Expose a config flag on the target algorithm; keep current default behaviour unchanged.
- Add tests for archive logic and a small algorithm smoke test using the new archive on ZDT1 or similar.

**Avoid**: Hard-wiring heavy dependencies; changing default semantics.

---

## 3. Extend tuning / AutoNSGA-II configuration space

**Goal**: Add new tunable parameters (e.g., mutation rate, archive type) to AutoNSGA-II or tuning spaces.

**Context**: `src/vamos/engine/tuning/racing/` for search spaces and racing tuners; `src/vamos/engine/algorithm/` config bindings; examples/notebooks under `examples/` or `notebooks/`.

**Steps**
- Extend the configuration space with ranges/types (continuous/categorical) and sensible defaults.
- Wire parameters into algorithm construction so tuning outputs actually change behaviour.
- Keep reproducibility (seeding) intact.
- Update examples/tests to cover the new parameters.

**Avoid**: Silent default changes without documentation.

---

## 4. Add a new real-coded variation operator

**Goal**: Introduce a new crossover/mutation operator for continuous variables.

**Context**: `src/vamos/engine/operators/real/`; operator registries/factories used by algorithms and tuning; tests under `tests/operators/real/`.

**Steps**
- Use an existing operator (e.g., SBX, BLX-alpha, polynomial mutation) as a template.
- Implement vectorized behaviour with bounds handling; keep the operator stateless aside from parameters.
- Export in `__init__.py` and register anywhere needed for configs.
- Add tests for shapes, bounds, and invariants; optional smoke test inside an algorithm.

**Avoid**: Changing semantics of existing operators unless explicitly requested.

---

## 5. Add a new benchmark or real-world problem

**Goal**: Implement and register a new optimization problem compatible with existing algorithms.

**Context**: `src/vamos/foundation/problem/` (benchmark and real_world subpackages and registry); CLI expects registered names.

**Steps**
- Define dimension, bounds, objectives, constraints; implement `evaluate` / `evaluate_population`.
- Register the problem with a canonical ID so `python -m vamos.experiment.cli.main --problem <id>` works.
- Add tests for shapes/finite outputs and any reference points.
- Add a small example snippet or script showing NSGA-II on the new problem with a tiny budget (user-facing imports via `vamos.problems` / `vamos.algorithms` / `vamos.api`).

**Avoid**: Long-running evaluations without documenting the cost.

---

## 6. Create a study / experiment pipeline

**Goal**: Define a reproducible study (problem x algorithm x seeds) wired to the CLI/study runner.

**Context**: `src/vamos/experiment/study/`, central orchestration in `src/vamos/experiment/runner.py`, and CLI under `src/vamos/experiment/cli/`; config-driven runs via `--config path.yaml`.

**Steps**
- Define study configuration (problems, algorithms, budgets, seeds, output directory).
- Loop over seeds/problems/algorithms using the central runner/study runner; write outputs under the standard layout in `results/` (or configured output_root).
- Expose a CLI flag/subcommand (e.g., via `vamos.experiment.cli.main`) to launch the study.
- Add a small smoke test that checks outputs (front files/metadata) are created.

**Avoid**: Hardcoded absolute paths or embedding heavy analytics in the core loop.

---

## 7. Improve logging, metadata, and analysis helpers

**Goal**: Enhance metadata/results handling and add helpers without breaking consumers.

**Context**: `src/vamos/foundation/core/metadata/`, `src/vamos/foundation/core/io/`, user-facing helpers under `src/vamos/ux/analysis/`, `src/vamos/ux/analytics/`, or `src/vamos/ux/visualization/`.

**Steps**
- Add additive metadata fields (e.g., backend, git commit if available) with safe defaults.
- Keep writing to the standard layout (`<output_root>/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/`).
- Provide loaders/aggregators (DataFrame if pandas already present via extras) that read this schema (see `vamos.ux.analysis.results`).
- Update tests to cover new metadata fields and helper functions.

**Avoid**: Schema-breaking changes or large binary blobs in metadata.

---

## 8. Add to the benchmarking CLI (`vamos-benchmark`)

**Goal**: Extend benchmark suites or algorithms used by `vamos-benchmark`.

**Context**: Benchmark definitions under `src/vamos/experiment/benchmark/` (suites, reference fronts, reporting); CLI entry `vamos-benchmark`.

**Steps**
- Locate suite definitions (e.g., `ZDT_small`); add new problems/algorithms with sensible budgets.
- Ensure outputs (raw runs, summary CSVs, LaTeX/plots) remain compatible.
- Add/update tests or smoke runs for the new suite with minimal budgets.
- Document example invocation, e.g., `vamos-benchmark --suite <suite> --algorithms nsgaii moead --output report/`.

**Avoid**: Changing default suites without noting compatibility impact.

---

## 9. Extend diagnostics / self-check tooling

**Goal**: Improve `vamos-self-check` / `vamos.experiment.diagnostics.self_check` to validate installs and basic runs.

**Context**: `src/vamos/experiment/diagnostics/` (or similar module); CLI entry `vamos-self-check`.

**Steps**
- Identify existing checks (deps, minimal algorithm runs).
- Add new fast checks (e.g., verify extras/backends availability, small ZDT1 run with reduced budget).
- Keep runtime short; guard optional deps so failures are informative.
- Add/update tests for the new diagnostics paths.

**Avoid**: Lengthy computations or mandatory heavy extras.

---

## Prompt patterns for AI agents (helper)

Use `AGENTS_codex_prompts.md` for ready-to-paste prompts aligned with this task list and README.

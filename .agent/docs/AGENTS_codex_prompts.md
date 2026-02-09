# AGENTS_codex_prompts.md

Ready-to-paste prompts for AI coding agents (e.g., Code, Copilot, GPT) working on **VAMOS**.

Assumptions:
- `AGENTS.md` and `AGENTS_tasks.md` are in the repo root.
- The agent can read project files before writing code.
- **User-friendliness is paramount**: Always prefer the clean public API `from vamos import ...` for user-facing code. Layered imports (`vamos.foundation.*`, `vamos.engine.*`, `vamos.experiment.*`, `vamos.ux.*`) are for contributor/internal work only.

### User-Friendly Import Pattern

```python
# USER-FACING CODE (examples, notebooks, documentation)
from vamos import optimize, make_problem_selection
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1, FeatureSelectionProblem, HyperparameterTuningProblem
from vamos.ux.api import plot_pareto_front_2d, weighted_sum_scores
from vamos.engine.tuning.api import ParamSpace, RandomSearchTuner, RacingTuner

# INTERNAL/CONTRIBUTOR CODE ONLY
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.registry import PROBLEM_SPECS
```

---

## 0. Global context loading (run once per session)

**Prompt 0 - Load and summarise VAMOS rules**

> You are an expert Python engineer working on VAMOS (Vectorized Architecture for Multiobjective Optimization Studies).  
> 1) Read and summarise `AGENTS.md` and `AGENTS_tasks.md`.  
> 2) Produce a concise checklist covering: (a) **user-friendliness** (clean public API, sensible defaults), (b) architectural concepts (problem, kernel, algorithm, operators, tuning, study, archive, benchmarking/diagnostics), (c) coding conventions and constraints (type hints, vectorization, tests, dependency/performance rules), (d) rules for adding new features and updating tests/docs.  
> 3) Confirm you will follow all of these rules for subsequent changes.  
> Output: a short bullet list of rules you will respect and a 3-5 sentence architecture summary.

---

## 1. Run a baseline experiment on ZDT1

**Prompt 1 - Minimal ZDT1 baseline + smoke test**

> You are in the VAMOS repo. You have read `AGENTS.md` and `AGENTS_tasks.md`.  
> Goal: create a minimal ZDT1 baseline using NSGA-II via `python -m vamos.experiment.cli.main` plus a pytest smoke test.  
> Tasks:  
> - Locate ZDT1 under `src/vamos/foundation/problem/` and NSGA-II under `src/vamos/engine/algorithm/`.  
> - Add a tiny example (script or notebook-style module) under `examples/` or `notebooks/` that runs `python -m vamos.experiment.cli.main --problem zdt1 --max-evaluations <small>` (and optionally `--engine moocore`). Keep budgets CI-friendly.  
> - Add a smoke test (e.g., `tests/study/test_zdt1_baseline.py`) asserting the final front is non-empty and finite.  
> - **For user-facing snippets, use the public API**: `from vamos import optimize; from vamos.algorithms import NSGAIIConfig; from vamos.problems import ZDT1`.  
> Follow all dependency/style rules; show diffs/new files and how to run the example/test.

---

## 2. Add or modify an external archive

**Prompt 2 - Hypervolume-based external archive option**

> You are in the VAMOS repo. Rules from `AGENTS.md` / `AGENTS_tasks.md` apply.  
> Goal: implement a hypervolume-based external archive and make it selectable in an algorithm (e.g., NSGA-II) without changing defaults.  
> Tasks:  
> - Find archive abstractions and hypervolume helpers (e.g., under `src/vamos/foundation/eval/`, `metrics/`, `algorithm/`, or kernels/backends).  
> - Implement `HypervolumeContributionArchive` (capacity handling, `add`/`extend`) with graceful fallback if hypervolume backends are missing.  
> - Add an algorithm config flag (e.g., `archive_type: none|crowding|hypervolume`) keeping current defaults unchanged.  
> - Add unit tests for the archive and a tiny ZDT1 smoke run using `archive_type="hypervolume"`.  
> Provide diffs/new files and a short usage snippet showing how to enable the archive.

---

## 3. Extend tuning / AutoNSGA-II configuration space

**Prompt 3 - Extend tuning search space and binding**

> You are in the VAMOS repo and will follow `AGENTS.md` / `AGENTS_tasks.md`.  
> Goal: extend the AutoNSGA-II (or equivalent) tuning space with new parameters (e.g., `mutation_rate`, `crossover_eta`, `archive_type`).  
> Tasks:  
> - Locate search space definitions under `src/vamos/engine/tuning/racing/` plus algorithm config binding under `src/vamos/engine/algorithm/`.  
> - Add ranges/types with sensible defaults; ensure parameters are passed into algorithm construction.  
> - Keep seeding/reproducibility intact.  
> - Update examples/tests so each new parameter is exercised on a tiny ZDT1 setup.  
> Output diffs/new files and an example config snippet.

---

## 4. Add a new real-coded variation operator

**Prompt 4 - Implement a real-coded crossover/mutation**

> You are in the VAMOS repo. Follow `AGENTS.md` and `AGENTS_tasks.md`.  
> Goal: add a new real-coded operator `<Name>` under `src/vamos/engine/operators/real/`.  
> Tasks:  
> - Use existing operators (SBX, BLX-alpha, polynomial mutation) as patterns.  
> - Implement vectorized behaviour with bounds handling; keep it stateless aside from parameters.  
> - Export in `__init__.py` and register anywhere configs expect it.  
> - Add tests under `tests/operators/real/test_<name>.py` for shapes, bounds, invariants; optionally a small algorithm smoke run.  
> Provide new files/diffs and a short snippet showing how to select the operator in an algorithm config.

---

## 5. Add a new benchmark or real-world problem

**Prompt 5 - Implement and register `<ProblemName>`**

> You are in the VAMOS repo. Rules from `AGENTS.md` / `AGENTS_tasks.md` apply.  
> Goal: add a new problem `<ProblemName>` that fits the Problem API and registry.  
> Tasks:  
> - Implement the problem under `src/vamos/foundation/problem/benchmark/` or `real_world/` with dimension, bounds, objectives, constraints, and `evaluate` / `evaluate_population`.  
> - Register it so `python -m vamos.experiment.cli.main --problem <id>` works.  
> - Add tests for shapes/finite outputs (and reference points if applicable).  
> - Add an example snippet or tiny script showing NSGA-II on the new problem with a small budget.  
> - If editing the paper benchmarking notebook (`notebooks/90_paper_benchmarking.ipynb`), keep the SAES-style critical distance plot toggle (`CD_STYLE`) intact.  
> Output code/tests/registry diffs and a brief description.

---

## 6. Create a study / experiment pipeline

**Prompt 6 - Reproducible study via CLI/study runner**

> You are in the VAMOS repo. Follow `AGENTS.md` / `AGENTS_tasks.md`.  
> Goal: define a study (problem x algorithm x seeds) with CLI wiring.  
> Tasks:  
> - Use `src/vamos/experiment/study/`, the central orchestration in `src/vamos/experiment/runner.py`, and CLI entry points to set up a study runnable via `python -m vamos.experiment.cli.main --config path.yaml` or a new flag.  
> - Implement looping over seeds/problems/algorithms; write outputs to the standard layout under `results/` (`<PROBLEM>/<algorithm>/<engine>/seed_<seed>/`).  
> - Add a smoke test with tiny budgets that asserts expected output files exist.  
> - Document example CLI usage.  
> Provide diffs/new files and test instructions.

---

## 7. Improve logging, metadata, and analysis helpers

**Prompt 7 - Add metadata fields and loader**

> You are in the VAMOS repo. Rules apply.  
> Goal: enhance run metadata and add a helper to load/aggregate it without breaking consumers.  
> Tasks:  
> - Update metadata/IO under `src/vamos/foundation/core/metadata/` and `core/io/` with additive fields (e.g., backend, git commit if available) and safe defaults.  
> - Keep outputs in the standard layout (`<output_root>/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/`). Add a loader/aggregator under `src/vamos/ux/analysis/` or `src/vamos/ux/analytics/` (DataFrame if pandas is already an installed extra) that reads this schema (see `vamos.ux.analysis.results`).  
> - Add tests for round-trip metadata and the loader using a synthetic minimal study directory.  
> Provide diffs/new files and a usage snippet.

---

## 8. Extend the benchmarking CLI (`vamos bench`)

**Prompt 8 - Add a benchmark suite entry**

> You are in the VAMOS repo. Follow the rules.  
> Goal: extend `vamos bench` with a new or updated suite.  
> Tasks:  
> - Locate suite/problem/algorithm definitions under `src/vamos/experiment/benchmark/` and any reference fronts.  
> - Add or adjust a suite (problems, budgets, algorithms) keeping output schema (raw runs, summary CSVs, LaTeX/plots) compatible.  
> - Add a smoke test (tiny budgets) that runs `vamos bench --suite <suite> --algorithms ... --output report/`.  
> - Document the new invocation in a short note or example.  
> Provide diffs/new files and the sample CLI command.

---

## 9. Extend diagnostics / self-check tooling

**Prompt 9 - Enhance `vamos check`**

> You are in the VAMOS repo. Follow the rules.  
> Goal: improve diagnostics (`vamos.experiment.diagnostics.self_check` / `vamos check`) to validate installs/backends quickly.  
> Tasks:  
> - Inspect existing diagnostics under `src/vamos/experiment/diagnostics/`.  
> - Add fast checks (e.g., optional backend availability, tiny ZDT1 run) guarded so missing extras produce clear messages.  
> - Keep runtime short; ensure defaults still pass in minimal installs.  
> - Add/update tests for the new checks.  
> Provide diffs/new files and the command to run the updated self-check.

---

## 10. Work with VAMOS Studio

**Prompt 10 - Update Studio experience**

> You are in the VAMOS repo. Rules apply.  
> Goal: extend the Studio (`vamos studio`) experience (e.g., add a new panel or preference scoring helper) while keeping defaults intact.  
> Tasks:  
> - Locate Studio code (e.g., `src/vamos/ux/studio/`) and how it reads study data from `results/`.  
> - Add the new feature (panel/helper) with clear defaults; guard optional deps required by the `studio` extra.  
> - Provide a minimal example dataset or instructions so `vamos studio --study-dir results` exercises the feature.  
> - Add tests or a lightweight integration check if feasible.  
> Output diffs/new files and usage notes.

---

## 11. Align tests and markers

**Prompt 11 - Add/adjust tests with markers**

> You are in the VAMOS repo. Follow the rules.  
> Goal: add or adjust tests while keeping the layered layout and markers consistent.  
> Tasks:  
> - Place new tests under the appropriate layer folder (`tests/foundation`, `tests/engine`, `tests/experiment`, `tests/ux`, or `tests/integration`).  
> - Apply markers: `smoke` for fast critical checks, `slow` for heavy runs, `backends` (or `numba`/`moocore`) for optional backends, `notebooks`/`examples` when relevant.  
> - Update `pyproject.toml` markers if new ones are introduced and mention any CI invocation changes (e.g., `pytest -m "smoke"`).  
> - Provide diffs and commands for running the relevant subsets.  

---

## 12. QA/self-review prompt

> You have just modified several files in the VAMOS repository.
>
> 1. List all functions/classes whose public API changed (signatures, default values, behaviour).
> 2. For each change, check whether there is at least one test that covers the new or changed behaviour.
>    - If not, propose specific new tests to add (paths + test names + what they assert).
> 3. Check that all new imports respect the dependency and performance contract in `AGENTS.md`.
> 4. Check that all new modules are placed under the appropriate package (`algorithm`, `kernel`, `problem`, `tuning`, `study`, etc.).
> 5. Suggest any small follow-up cleanup (dead code, duplicated helpers, missing docstrings) that would improve maintainability without large refactors.
>
> Output a concise checklist and, if needed, patches for missing tests.

---

You can keep this file in the repo and copy any of these prompts directly into your coding agent when starting a new task on VAMOS. Each prompt aligns with README and the guidance in `AGENTS.md` and `AGENTS_tasks.md`.

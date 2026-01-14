# Problem Registry (foundation layer)

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


This file defines the rules for adding and maintaining problem specs without
creating new monoliths or cross-layer tangles.

## Part I — Where to add new problem specs (canonical workflow)

Canonical layout:
- `registry/common.py` — `ProblemSpec` + shared helpers (keep focused)
- `registry/families/<family>.py` — canonical spec definitions (ZDT/DTLZ/WFG/etc.)
- `registry/specs.py` — thin aggregator only (no business logic)

Hard rule: **Never add new specs directly to `registry/specs.py`.**

Minimal example (add a spec to the correct family module):

```python
# src/vamos/foundation/problem/registry/families/misc.py
from ..common import ProblemSpec
from ...my_family import MyProblem

SPECS["my_problem"] = ProblemSpec(
    key="my_problem",
    label="My Problem",
    default_n_var=2,
    default_n_obj=2,
    allow_n_obj_override=False,
    description="Short description of the landscape.",
    factory=lambda n_var, _n_obj: MyProblem(n_var=n_var),
    encoding="continuous",
)
```

`registry/specs.py` will aggregate all family `SPECS` into `PROBLEM_SPECS`,
so your spec becomes available automatically.

## Part II — Architectural Health Guardrails (no monoliths, no tangling)

Monolith prevention:
- Keep each family module cohesive. If a family file grows beyond ~250 non-blank LOC,
  split by sub-family or theme.
- `registry/specs.py` must remain an aggregator only (<200 LOC). No factories, no logic.
- Use `registry/common.py` for shared helpers instead of copy/paste, but keep it focused.

Cross-reference prevention:
- Family modules MUST NOT import each other; shared utilities go in `common.py`.
- Registry modules MUST NOT import engine/experiment/ux.
- Do not add "convenience" imports that cross layers; use canonical facades instead.

Layering rules (project-wide):
- foundation must not import engine/ux/experiment
- engine may depend on foundation and hooks only
- experiment and ux may depend on foundation and engine

Public surface discipline:
- `vamos/__init__.py` stays minimal.
- Use facades: `vamos.api`, `vamos.algorithms`, `vamos.ux.api`,
  `vamos.experiment.quick`.

## Part III — Health checks before opening a PR

Checklist:
- Architecture boundaries:
  - `.\.venv\Scripts\python.exe -m pytest -q tests/architecture/test_layer_boundaries.py`
- Foundation tests:
  - `.\.venv\Scripts\python.exe -m pytest -q tests/foundation -q`
- Full suite:
  - `.\.venv\Scripts\python.exe -m pytest -q`
- Optional static checks (if installed/configured):
  - `ruff check src tests`
  - `mypy src`

If you violate the rules:
- Split a growing family module into smaller files.
- Move shared helpers to `registry/common.py`.
- Remove cross-layer imports and use the proper facades instead.

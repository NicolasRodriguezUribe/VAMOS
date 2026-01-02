# ADR 0003: Optional Dependencies Policy

## Status
Accepted

## Context
Optional dependencies (benchmarks, plotting, UI, autodiff) should not be required
to import or use core optimization logic.

## Decision
- Core dependencies live in `[project].dependencies`.
- Heavy or optional libraries must appear only under `[project.optional-dependencies]`.
- Optional imports must be lazy or guarded with `try/except ImportError`.
- `experiment/external` is the integration boundary for benchmark backends.
- `ux/studio` is the only place `streamlit` is allowed.

## Consequences
- Core installs remain lightweight.
- Optional features are clearly scoped and testable.
- A dependency policy gate prevents regressions.

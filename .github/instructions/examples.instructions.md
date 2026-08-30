---
applyTo: "examples/**/*.py"
description: Runnable example deltas for VAMOS
---

Follow [/AGENTS.md](/AGENTS.md) and use public facades in published examples.

- Make the script self-contained, seeded, and runnable from the repository root with a modest evaluation budget.
- Put reusable behavior in `src/vamos/` with tests; examples demonstrate rather than define APIs.
- Add machine-executed documentation examples to `tests/docs/smoke_manifest.py` when they establish a supported workflow.
- State optional extras in the module docstring and avoid machine-specific output paths.

```agent-docs
path: AGENTS.md
path: tests/docs/smoke_manifest.py
```

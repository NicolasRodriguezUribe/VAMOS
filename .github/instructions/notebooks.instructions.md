---
applyTo: "notebooks/**/*.ipynb"
description: Notebook-specific deltas for VAMOS
---

Follow [/AGENTS.md](/AGENTS.md) and use public facades.

- Keep setup explicit, seeds fixed, and default budgets suitable for a smoke run.
- Move reusable logic into `src/vamos/` and cover it with Python tests.
- Clear transient errors, widget state, secrets, and bulky generated output before committing.
- Store durable figures/results outside the notebook and use repository-relative paths.
- Run the notebook smoke gate when changing an executed learning path.

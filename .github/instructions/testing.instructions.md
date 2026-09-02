---
applyTo: "tests/**/*.py"
description: Pytest-specific deltas for VAMOS
---

Follow [/AGENTS.md](/AGENTS.md) and [Testing and validation](/docs/dev/testing.md).

- Place coverage beside the owning layer and name tests after observable behavior.
- Keep unit budgets tiny; seed stochastic code and assert invariants rather than incidental random samples.
- Use `tmp_path` for outputs and `pytest.importorskip` or existing markers for optional dependencies.
- Tests must not require network access, user credentials, global RNG state, or a pre-existing local environment directory.
- Add subprocess coverage for CLI parsing, exit status, and artifact publication boundaries.

```agent-docs
path: AGENTS.md
path: docs/dev/testing.md
```

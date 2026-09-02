---
applyTo: "**/*.yaml,**/*.yml,**/*.json"
description: Machine-readable configuration deltas for VAMOS
---

Follow [/AGENTS.md](/AGENTS.md). Preserve the schema and key style of the nearest validated example; do not infer keys from prose or another subsystem.

- Keep JSON strict and YAML portable: no executable tags, implicit Python objects, or environment-specific absolute paths.
- For an experiment specification, validate with `vamos --config <path> --validate-config` before execution.
- For a committed fixture, keep values deterministic, budgets small, and paths repository-relative.
- Do not hand-edit generated manifests or reports as configuration templates.

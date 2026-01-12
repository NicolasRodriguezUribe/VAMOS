# ADR 0001: Layering and Facades

## Status
Accepted

## Context
VAMOS has multiple layers (foundation, engine, experiment, ux) and a small set of public facades.
Uncontrolled cross-layer imports lead to architectural erosion and tighter coupling.

## Decision
- Enforce strict layer boundaries:
  - foundation must not import engine/ux/experiment.
  - engine may depend on foundation and hooks only.
  - experiment and ux may depend on foundation and engine.
- Public entrypoints must be explicit and minimal:
  - `vamos.api`, `vamos.engine.api`, `vamos.ux.api`.
  - `vamos/__init__.py` remains a small facade.

## Consequences
- Core modules remain dependency-clean and swappable.
- Public API changes are intentional and visible.
- Layer boundary violations are caught by automated gates.

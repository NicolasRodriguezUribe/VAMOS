# ADR 0001: Layering and Facades

## Status
Accepted

## Context
VAMOS has multiple layers (foundation, engine, experiment, ux, assist, resources)
and a small set of public facades.
Uncontrolled cross-layer imports lead to architectural erosion and tighter coupling.

## Decision
- Enforce strict layer boundaries:
  - foundation may depend on foundation/resources only.
  - engine may depend on engine/foundation/resources.
  - experiment may depend on experiment/foundation/engine/ux/assist/resources.
  - ux may depend on ux/foundation/engine/resources.
  - assist may depend on assist/foundation/engine/experiment/resources.
  - resources must not import other VAMOS layers.
- Public entrypoints must be explicit and minimal:
  - `vamos.api`, `vamos.algorithms`, `vamos.problems`, `vamos.ux.api`.
  - `vamos/__init__.py` remains a small facade.

## Consequences
- Core modules remain dependency-clean and swappable.
- Public API changes are intentional and visible.
- Layer boundary violations are caught by automated gates.

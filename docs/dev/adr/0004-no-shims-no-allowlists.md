# ADR 0004: No Shims, No Allowlists

## Status
Accepted

## Context
Compatibility shims and allowlists mask architectural issues and delay fixes.
They also complicate the codebase and erode enforcement of guardrails.

## Decision
- Do not add compatibility shims or deprecation wrappers.
- Do not introduce allowlists for architecture gates.
- Fix violations at the source and update call sites.

## Consequences
- Cleaner dependency graph and clearer ownership.
- Enforcement remains strict and predictable.

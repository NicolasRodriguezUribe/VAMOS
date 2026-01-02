# ADR 0002: Import-Time Purity

## Status
Accepted

## Context
Import-time side effects make modules slow to load, brittle in tooling, and hard to reuse.
They also violate optional dependency and logging policies.

## Decision
At import time, allow only:
- imports
- constant/type alias assignments (no calls)
- function/class definitions
- decorators
- `if TYPE_CHECKING:` blocks
- `if __name__ == "__main__":` blocks

All executable calls (file IO, dynamic imports, environment reads, initialization) must live
inside functions or CLI entrypoints.

## Consequences
- Safer imports for users and tooling.
- Optional dependencies load only when invoked.
- Import-time purity is enforced by a strict AST gate with no allowlist.

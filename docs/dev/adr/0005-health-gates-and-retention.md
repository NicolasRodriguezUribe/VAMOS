# ADR 0005: Health Gates and External Validation Evidence

## Status
Accepted; the original in-repository report-retention decision is superseded.

## Context
Architecture regressions and generated repository clutter are expensive to
unwind. Fast-fail gates and explicit external evidence ownership keep the repo
healthy long term.

## Decision
- CI runs a fast-fail suite of architecture health gates before full tests.
- Local developers use `python tools/health.py` for the canonical local fast-fail suite. CI has its own matrix and coverage scope; individual shared gates must use the same command and arguments in both places.
- `python tools/check_repository_hygiene.py` is a shared health, CI and release gate.
- Raw audits, Goal handoffs and validation logs are external evidence or CI artifacts; no canonical audit copy is retained at root.
- Public conclusions are edited into maintained documentation.
- Generated reports and publication builds use ignored output directories.
- The detailed current contract and exception process are defined in `docs/dev/repository_hygiene.md`.

## Consequences
- Problems are detected early and deterministically.
- The product tree does not double as local evidence storage.
- External evidence remains hashable and recoverable without becoming package or source-distribution content.

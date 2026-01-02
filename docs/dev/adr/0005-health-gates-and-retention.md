# ADR 0005: Health Gates and Report Retention

## Status
Accepted

## Context
Architecture regressions and documentation bloat are expensive to unwind.
Fast-fail gates and strict report retention keep the repo healthy long term.

## Decision
- CI runs a fast-fail suite of architecture health gates before full tests.
- Local developers use `python tools/health.py` for the same suite.
- Only one canonical audit is kept at repo root: `final_audit_latest.md`.
- Reports and artifacts are capped (newest five, archive capped, size budget).

## Consequences
- Problems are detected early and deterministically.
- Audit history stays bounded and reviewable.

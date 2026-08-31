# ADR 0007: Canonical Typing Gates

## Status

Accepted

## Context

VAMOS had three incompatible typing signals: a selected CI scope, an unenforced total-count file, and full-source mypy in local health. They used no single reproducible interpretation of passing typing.

## Decision

- `tools/typecheck.py` is the sole typecheck entry point.
- Python 3.12, compiled mypy 1.15.0, typing-extensions 4.16.0, `pyproject.toml`, and `constraints/ci.txt` define the supported environment.
- Strict typing covers at least the former CI paths and requires zero diagnostics.
- Full development typing enforces one structured multiset baseline. Fingerprints exclude source locations but include path, error code, and normalized message; multiplicity cannot increase.
- Changed production files must contain no baseline debt.
- Resolved fingerprints are removed from the baseline in the same change.
- Release typing always checks all production source and requires zero diagnostics.
- Health and CI run the same strict and full commands. Release workflows run the release command.

## Consequences

Development health can pass while accurately reporting remaining structured debt, but it cannot hide a new diagnostic or stale allowance. Release remains blocked until full-source typing reaches zero. Environment or configuration changes require explicit baseline review.

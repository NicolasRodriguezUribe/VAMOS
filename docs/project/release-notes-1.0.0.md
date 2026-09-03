# VAMOS 1.0.0 release notes

Released 2026-09-03.

VAMOS 1.0.0 is the first official public release and the first compatibility
baseline. Earlier version strings, tags, and artifacts were internal
pre-public development markers, not public releases or compatibility promises.

## What is included

- A stable optimization facade with built-in problems and nine multi-objective
  algorithms.
- NumPy as the deterministic reference backend, with optional Numba kernel and
  MooCore indicator acceleration.
- Strict evaluation-budget and explicit seed/backend behavior.
- A canonical run lifecycle: save, load, inspect, verify, and exact
  same-environment replay for reconstructable built-ins.
- A canonical, single-owner durable study lifecycle: plan, create, run,
  inspect, summarize, cancel, resume, and retry.
- Versioned run and study schemas, bounded readers, integrity evidence, and
  stable CLI JSON envelopes.
- Installation, quickstart, run/replay, studies, Studio trust-boundary,
  stability, and limitation guides.

## Security boundary

Loading, inspecting, and verifying artifacts are data-only operations. Replay
is separate and explicit. Studio-generated Python is never a sandboxed input:
it runs with the current operating-system user's permissions only after the
user reviews the current code revision and opts into trusted local execution.
Studio listens on loopback by default; remote binding requires an explicit
command-line opt-in and emits a warning.

## Compatibility

The stable Python API, stable CLI commands, configuration fields, and public
artifact schemas named in the [stability policy](stability-and-versioning.md)
follow semantic versioning from 1.0.0 onward. Experimental and internal
surfaces are outside that commitment.

See [Known limitations](known-limitations.md) before adopting optional
backends, replay, studies, tuning, plugins, or Studio.

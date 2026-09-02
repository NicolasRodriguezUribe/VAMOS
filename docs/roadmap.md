# Roadmap

This roadmap records direction without delivery dates. It is not a
compatibility promise; only the
[stability policy](project/stability-and-versioning.md) defines public
compatibility commitments. Completed 1.0.0 work is recorded in the changelog,
not here.

## Study execution

| Item | Status | Motivation | Measurable completion criterion | Dependency |
| --- | --- | --- | --- | --- |
| Local study lock | Exploring | Prevent accidental competing mutators. | Two local processes cannot both acquire ownership; stale-lock recovery passes crash tests. | Portable lock semantics. |
| Durable task claims | Exploring | Make work assignment explicit before adding workers. | Every executable task transition carries one validated owner claim and rejects duplicate ownership. | Lock and journal design. |
| Leases | Planned | Bound ownership after worker loss. | Expired leases are reclaimed without allowing two valid commits for one attempt. | Durable task claims and a clock-skew policy. |
| Heartbeat | Planned | Distinguish slow work from abandoned ownership. | Fault-injection tests detect a stopped owner within a configured bound without disrupting a live owner. | Lease semantics. |
| Fencing | Planned | Reject writes from superseded owners. | A stale owner token cannot append events, checkpoints, attempts, or run references after reassignment. | Monotonic ownership epochs. |
| Bounded local worker pool | Planned | Use local CPU capacity without unbounded processes. | A configured worker limit is never exceeded and crash/retry tests preserve deterministic task identity. | Locks, claims, leases, heartbeat, and fencing. |
| Distributed/cross-host studies | Planned | Scale durable matrices beyond one machine. | A documented protocol passes duplicate-delivery, partition, reconnect, and recovery acceptance tests. | Stable local worker and ownership protocol. |

## Analysis

| Item | Status | Motivation | Measurable completion criterion | Dependency |
| --- | --- | --- | --- | --- |
| Statistical comparison | Exploring | Turn verified study outputs into attributable comparisons. | A versioned report records inputs, missingness, aggregation, and paired blocks for every statistic. | Stable study-to-run provenance. |
| Effect sizes | Planned | Report practical magnitude alongside significance. | Supported effect sizes have formula references, direction conventions, and independent numeric fixtures. | Statistical report schema. |
| Hypothesis tests | Planned | Make assumptions and multiplicity handling explicit. | Each test records hypotheses, assumptions, correction method, alpha, and deterministic fixture results. | Statistical report schema. |
| Attainment analysis | Planned | Compare stochastic fronts beyond scalar indicators. | Attainment surfaces reproduce reference fixtures within declared numerical tolerances. | Comparable objective scaling and verified fronts. |
| Publication reporting | Planned | Export reviewable tables and plots with provenance. | Every exported value links to a report field and ultimately to verified run IDs. | Stable analysis report. |
| Run checkpoints | Exploring | Support analysis of progress without weakening final-run integrity. | A versioned checkpoint format has bounded readers, integrity rules, and deterministic progress fixtures. | Public schema-evolution design. |

## Framework architecture

| Item | Status | Motivation | Measurable completion criterion | Dependency |
| --- | --- | --- | --- | --- |
| Declarative component descriptors | Exploring | Reconstruct components without importing arbitrary recorded code. | Built-ins round-trip through bounded JSON descriptors with schema and negative security tests. | Component identity model. |
| Machine-readable component catalog | Planned | Let tools discover verified capabilities. | The catalog reports stable IDs, encodings, dimensions, backends, and config schemas and matches registry tests. | Declarative descriptors. |
| Plugin-contract stabilization | Planned | Give third-party components an explicit compatibility boundary. | A versioned contract passes isolated conformance fixtures and documents deprecation rules. | Descriptors and catalog. |
| Backend-equivalence specification | Exploring | State which numerical comparisons are scientifically meaningful. | Each stable kernel has documented determinism/tolerance semantics and cross-backend oracle tests. | Independent numerical references. |

## Quality

| Item | Status | Motivation | Measurable completion criterion | Dependency |
| --- | --- | --- | --- | --- |
| Full-source mypy zero | In progress | Remove the frozen typing debt outside the strict/stable scopes. | Canonical mypy reports zero diagnostics for all `src/vamos` files and the ratchet is retired. | Incremental module cleanup without API churn. |
| Branch-coverage expansion | Planned | Exercise failure and recovery paths, not only statements. | A reviewed branch threshold covers critical run/study/security modules with documented exclusions only. | Stable, non-flaky coverage environment. |
| Broader notebook execution | Planned | Keep long-form learning material runnable. | Every primary notebook executes from a clean supported environment within a declared resource budget. | Curated notebook tiers and optional dependencies. |
| AGE-MOEA/RVEA/SMPSO oracles | Planned | Strengthen independent algorithmic evidence. | Each algorithm passes published small-case or independent implementation fixtures across multiple seeds. | Vetted reference data and tolerances. |
| Performance-regression framework | Exploring | Detect meaningful regressions without brittle timing claims. | Pinned benchmarks report distributions and flag preregistered effect-size thresholds on controlled runners. | Dedicated runners and baseline governance. |

## Artifacts and replay

| Item | Status | Motivation | Measurable completion criterion | Dependency |
| --- | --- | --- | --- | --- |
| Artifact authenticity/signing | Planned | Distinguish publisher identity from hash-based tamper detection. | Signed release and artifact verification succeeds against documented trust roots and fails for altered inputs. | Key-management and rotation policy. |
| Future public-schema evolution | Exploring | Evolve artifacts without stranding public data. | Every schema change has compatibility rules, golden old/new fixtures, and a tested reader transition. | Schema governance. |
| Explicitly non-exact reruns | Planned | Permit useful reruns when exact compatibility is impossible without mislabeling them. | A separately named mode records all substitutions and never reports `exact`; fixtures cover each downgrade reason. | Compatibility taxonomy and descriptor model. |

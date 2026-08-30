# Experiment artifact contract

Experiment and paper tooling has one source-of-truth input: a canonical VAMOS
run directory containing `manifest.json`, `result.npz`, and `environment.json`.
The manifest declares `document_type = "vamos.run-manifest"` and
`schema_version = "1.0.0"`.

Collectors discover runs through `manifest.json`, load identity and execution
context with `vamos.load_run`, and load numerical arrays with
`vamos.load_result`. They must not infer algorithm, problem, backend, seed, or
configuration from directory names.

## Artifact classes

- `SOURCE_INPUT`: problem data, frozen reference points, campaign
  specifications, and other irreplaceable scientific inputs. Preserve these.
- `CANONICAL_RUN`: the immutable run directory written by VAMOS. Never rewrite
  it during analysis.
- `DERIVED_REGENERABLE_OUTPUT`: tidy CSV tables, statistics, plots, LaTeX, and
  samples generated from canonical runs. These are analysis products, not run
  persistence.
- `PUBLICATION_ARCHIVE`: retained evidence for a published result. It may be
  read by publication-specific tooling but is never accepted as a current run.

## Derived tidy tables

`experiments/scripts/canonical_runs.py` is the shared adapter used by active
collectors. A derived row includes the canonical run/task identities, resolved
algorithm/problem/backend/seed, population and termination settings, outcome
timing, result dimensions, and flattened outcome metrics. Objective summaries
are calculated from arrays returned by `load_result`.

Derived tables may be written under `artifacts/tidy/` or
`experiments/sample_outputs/`. Their filenames and columns are analysis
interfaces only and must never be treated as a loadable VAMOS run format.

## Validation

Use a tiny canonical fixture to test collector discovery, manifest field
extraction, array loading, and table generation. Publication-scale campaigns
are outside routine validation.

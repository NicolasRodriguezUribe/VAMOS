# Release verification

`tools/release_check.py` is the canonical, fail-closed VAMOS release gate. It
validates the source tree, stable contracts, typing policy, full test suite,
both documentation sites, distributions, installed wheel, dependencies, and
release evidence in one command:

```bash
python tools/release_check.py --version 1.0.0
```

Run it from a clean release branch with the development, documentation,
compute, and release tooling installed. Build and release tools are pinned and
audited separately from runtime dependencies:

```bash
python -m pip install -c constraints/ci.txt -e ".[dev,docs,compute,studio]"
python -m pip install -r release/requirements-build.txt
python -m pip install -r release/requirements-tools.txt
```

For an auditable candidate, bind the check to the intended identity and write
evidence outside the repository:

```bash
python tools/release_check.py \
  --version 1.0.0 \
  --expected-branch release/1.0.0 \
  --expected-commit <full-commit-sha> \
  --tag-state pre-normalization \
  --output-dir <new-empty-evidence-directory>
```

Use `--json` when a machine consumer requires exactly one JSON document on
standard output. `--list-checks` exposes the ordered check inventory. Evidence
directories are append-never: the checker refuses to overwrite a non-empty
directory.

## Frozen distributions

The candidate workflow builds one wheel and one sdist exactly once. Every
downstream platform downloads those same bytes. When validating an existing
candidate, pass its directory instead of rebuilding it:

```bash
python tools/release_check.py \
  --version 1.0.0 \
  --artifacts <directory-with-one-wheel-and-one-sdist> \
  --output-dir <new-empty-evidence-directory>
```

The evidence contains SHA-256 checksums, a CycloneDX SBOM, the installed
runtime lock, dependency-audit reports, artifact metadata/content inspection,
and provenance tied to the source commit. Publication consumes the frozen
workflow artifact and verifies every byte again; it never rebuilds from the
tag.

## Installed-wheel smoke

`tools/release_smoke.py` is designed to run from a clean, non-editable wheel
environment with no repository import path. Its full mode exercises public
imports, optimization, byte-exact replay, durable study execution, controlled
failure/resume/retry, relocation, and human and JSON CLI contracts while
blocking network access:

```bash
python tools/release_smoke.py --version 1.0.0 --mode full
```

`--mode core` omits the optional-backend failure scenario and is used by the
minimal and non-primary platform jobs. The hosted matrix covers Linux, Windows,
and macOS; minimum Python, primary Python, and the declared maximum tested
Python; a minimal wheel and the supported optional dependency set.

## Gate policy

Strict and stable typing must be clean. Full-source typing must match its exact
no-regression baseline; `full-zero` remains informational for VAMOS 1.0.0 and
does not claim that inherited typing debt is resolved. Runtime dependency
findings block the release. Build and release-tool audits are recorded
separately and surface findings as warnings for owner review; the pinned 1.0.0
tool sets are expected to audit clean.

The pre-normalization tag state is required until the archived internal tags
are removed. After normalization, use `--tag-state normalized`. Do not publish
if repository identity, artifact hashes, TestPyPI installation, or any critical
gate differs from the frozen candidate.

TestPyPI is published first using trusted publishing. Only after its exact
wheel installs and passes the full smoke may the same immutable wheel and sdist
be sent to PyPI and attached to the GitHub release with their checksums, SBOM,
manifest, provenance, stability contract, and known limitations.

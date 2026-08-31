# Testing and validation

Use the smallest test that proves a change while iterating, then run every higher tier required by `/AGENTS.md`. Tests are deterministic, offline, isolated from user directories, and explicit about optional dependencies.

## Test placement

- `tests/foundation/`: problems, evaluation, kernels, indicators, resources, and base contracts.
- `tests/engine/`: algorithms, operators, archives, hooks, hyperheuristics, and tuning.
- `tests/experiment/`: orchestration, CLI, studies, artifacts, and diagnostics.
- `tests/ux/`: analysis, visualization, and Studio behavior.
- `tests/integration/` and `tests/e2e/`: cross-layer and command workflows.
- `tests/architecture/`: dependency, import, formatting, packaging, API, and repository policy gates.
- `tests/docs/`: machine-executed learning and CLI documentation.

Use `tmp_path` for writes, seed stochastic code, and keep algorithm budgets tiny. Prefer exact invariants—shape, feasibility, evaluation count, deterministic replay, parity, error type—over fragile incidental samples. Mark optional/slow coverage with the existing markers and use `pytest.importorskip` when absence is valid.

## Canonical tiers

Targeted:

```bash
python -m pytest -q <nearest-test-files>
```

For agent/documentation changes:

```bash
python tools/check_agent_docs.py
python -m pytest -q tests/test_check_agent_docs.py tests/architecture/test_docs_and_workflows.py tests/docs
```

Full local validation:

```bash
python tools/health.py
python -m pytest -q
mkdocs build --strict
```

Release validation adds a global zero-error typecheck before packaging:

```bash
python tools/typecheck.py --scope release
python -m build
```

Then follow [Release Smoke Verification](../release_smoke.md) for the affected distribution path. `tools/health.py` is a local fast-fail suite; CI uses a separate platform/version matrix and coverage command. Their complete scopes differ. The agent-documentation check itself is intentionally identical in both. The strict and full typecheck commands are also identical between health and the dedicated CI typing job.

## Canonical typing

Install and verify the environment described in [Typing policy](typing.md), then run:

```bash
python tools/typecheck.py --scope strict
python tools/typecheck.py --scope full
```

Strict requires zero diagnostics over the protected path inventory. Full development typing passes only when the normalized diagnostic multiset exactly matches `typing/mypy-baseline.json` and changed production files contain no debt. A reduction makes the baseline stale and must be recorded in the same change. Release uses `--scope release`, ignores no debt, and requires global zero.

`tools/check_pre_release_remnants.py` owns the repository-wide semantic scan and shared discarded-token rules. `tools/check_agent_docs.py` imports those shared rules for instruction files and adds scope, adapter, link, declaration, duplicate-body, and agent-policy validation. Health and CI execute both checkers once; neither checker recursively invokes the other.

## Changed Python checks

Run Ruff, format, typing, compilation, and whitespace checks in proportion to the change:

```bash
python -m ruff check <changed-python-and-test-paths>
python -m ruff format --check <changed-python-and-test-paths>
python tools/typecheck.py --scope strict
python tools/typecheck.py --scope full
python -m compileall <changed-python-and-test-paths>
git diff --check
```

When a subprocess imports VAMOS, set `PYTHONPATH` to the intended worktree's `src` directory and verify `vamos.__file__` first. Do not use an editable install to claim a clean wheel smoke.

## Failure reporting

Record the exact command, exit code, and short result. Classify a failure as an implementation/instruction/checker/documentation defect, environment limitation, or pre-existing unrelated failure. Do not hide skipped gates or repair unrelated failures under the current task.

```agent-docs
path: tools/health.py
path: tests/test_check_agent_docs.py
path: tests/architecture/test_docs_and_workflows.py
path: tests/docs
path: docs/release_smoke.md
path: docs/dev/typing.md
path: pyproject.toml
path: tools/typecheck.py
path: typing/mypy-baseline.json
command: python tools/check_agent_docs.py
command: python -m pytest -q tests/test_check_agent_docs.py tests/architecture/test_docs_and_workflows.py tests/docs
command: python tools/health.py
command: python tools/typecheck.py --scope strict
command: python tools/typecheck.py --scope full
command: python tools/typecheck.py --scope release
command: python -m pytest -q
command: mkdocs build --strict
command: python -m build
```

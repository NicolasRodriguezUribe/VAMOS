# Pull Request Checklist

## Architecture health
- [ ] I read the ADRs (`docs/dev/adr/index.md`) relevant to this change.
- [ ] I ran the local health command: `python tools/health.py`.
- [ ] I updated `docs/dev/architecture_health.md` if I changed guardrails.

## Public API & dependencies
- [ ] Public API changes are intentional; snapshot updated via `python tools/update_public_api_snapshot.py` if needed.
- [ ] No optional/heavy dependencies were added to core `[project].dependencies`.

## Behavior & tests
- [ ] Runtime behavior is unchanged (or explicitly documented as a bugfix).
- [ ] Tests pass locally (`pytest -q`).

## Reports/retention
- [ ] New audit outputs (if any) are under `reports/<audit>_artifacts/`.
- [ ] `final_audit_latest.md` is updated only when a new audit report is added.

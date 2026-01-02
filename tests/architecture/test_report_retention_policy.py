from __future__ import annotations

import re
from pathlib import Path

MAX_AUDITS = 5
MAX_ARTIFACT_DIRS = 5
MAX_ARCHIVE_FILES = 20
MAX_MARKDOWN_BYTES = 15 * 1024 * 1024
LOG_PREFIXES = ("mypy", "ruff", "build_output", "full_pytest_summary", "gates_run")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _audit_number(path: Path) -> int:
    match = re.search(r"final_audit_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return -1


def test_report_retention_policy() -> None:
    repo_root = _repo_root()
    reports_root = repo_root / "reports"
    archive_root = reports_root / "archive"
    canonical = repo_root / "final_audit_latest.md"

    root_audits = sorted(repo_root.glob("final_audit_*.md"))
    extra_root_audits = [p for p in root_audits if p.name != "final_audit_latest.md"]
    if extra_root_audits:
        names = ", ".join(p.name for p in extra_root_audits)
        raise AssertionError(f"Only final_audit_latest.md is allowed at repo root. Move these into reports/: {names}")

    root_logs = sorted(p.name for p in repo_root.glob("*.txt") if p.name.startswith(LOG_PREFIXES))
    if root_logs:
        names = ", ".join(root_logs)
        raise AssertionError(
            f"Raw logs are not allowed at repo root. Move them under reports/<audit>_artifacts/ or reports/archive/. Found: {names}"
        )

    if not canonical.is_file():
        raise AssertionError("final_audit_latest.md is required at repo root.")

    audits = sorted(p for p in reports_root.glob("final_audit_*.md") if p.is_file())
    if len(audits) > MAX_AUDITS:
        names = ", ".join(p.name for p in audits)
        raise AssertionError(
            f"Too many final audit reports in reports/ ({len(audits)} > {MAX_AUDITS}). "
            f"Move older audits to reports/archive/. Found: {names}"
        )

    if not audits:
        raise AssertionError("No final_audit_*.md files found in reports/.")

    newest = max(audits, key=lambda p: (_audit_number(p), p.name))
    latest_text = canonical.read_text(encoding="utf-8")
    newest_text = newest.read_text(encoding="utf-8")
    if latest_text != newest_text:
        raise AssertionError(f"final_audit_latest.md must match the newest audit report content. Newest is {newest.name}.")

    artifact_dirs = sorted(p for p in reports_root.glob("final_audit_*_artifacts") if p.is_dir())
    if len(artifact_dirs) > MAX_ARTIFACT_DIRS:
        names = ", ".join(p.name for p in artifact_dirs)
        raise AssertionError(
            f"Too many audit artifact directories in reports/ ({len(artifact_dirs)} > {MAX_ARTIFACT_DIRS}). "
            f"Move older artifacts to reports/archive/. Found: {names}"
        )

    if archive_root.exists():
        archive_files = sorted(p for p in archive_root.rglob("*") if p.is_file())
        if len(archive_files) > MAX_ARCHIVE_FILES:
            raise AssertionError(
                f"reports/archive has {len(archive_files)} files (limit {MAX_ARCHIVE_FILES}). Remove or consolidate old audits."
            )

    stray_logs: list[str] = []
    for path in reports_root.rglob("*.txt"):
        rel = path.relative_to(reports_root).as_posix()
        if rel.startswith("archive/"):
            continue
        if "/final_audit_" in f"/{rel}" and "_artifacts/" in f"/{rel}":
            continue
        if path.name.startswith(LOG_PREFIXES):
            stray_logs.append(rel)

    if stray_logs:
        lines = "\n".join(f"- {item}" for item in sorted(stray_logs))
        raise AssertionError(f"Raw log files must live under reports/<audit>_artifacts/ or reports/archive/.\n{lines}")

    markdown_bytes = 0
    for path in reports_root.rglob("*.md"):
        rel = path.relative_to(reports_root).as_posix()
        if rel.startswith("archive/"):
            continue
        markdown_bytes += path.stat().st_size
    if markdown_bytes > MAX_MARKDOWN_BYTES:
        mb = markdown_bytes / (1024 * 1024)
        raise AssertionError(
            f"reports/ markdown size is {mb:.2f} MB (limit {MAX_MARKDOWN_BYTES / (1024 * 1024):.2f} MB). Archive or delete old reports."
        )

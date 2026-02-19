from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from shutil import which

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_PATH = Path(__file__).with_name("mypy_error_budget.json")
SUMMARY_RE = re.compile(r"Found (\d+) errors in (\d+) files")
ERROR_LINE_RE = re.compile(r"^(?P<path>[^:]+\.py):\d+:")


def _find_mypy_cmd() -> list[str]:
    mypy_exe = Path(sys.executable).with_name("mypy.exe")
    if mypy_exe.exists():
        return [str(mypy_exe)]
    path_mypy = which("mypy")
    if path_mypy:
        return [path_mypy]
    return [sys.executable, "-m", "mypy"]


def _run_mypy() -> tuple[int, str]:
    cmd = _find_mypy_cmd() + [
        "--config-file",
        "pyproject.toml",
        "src/vamos",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def _extract_error_count(output: str) -> int:
    match = SUMMARY_RE.search(output)
    if match:
        return int(match.group(1))
    return output.count(" error:")


def _top_offenders(output: str, limit: int = 5) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        match = ERROR_LINE_RE.match(line)
        if match:
            counts[match.group("path")] += 1
    return counts.most_common(limit)


def test_mypy_error_budget() -> None:
    if not BUDGET_PATH.exists():
        raise AssertionError(f"Missing mypy budget file: {BUDGET_PATH}")
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    max_errors = int(budget.get("max_errors", 0))

    returncode, output = _run_mypy()
    if not output.strip():
        raise AssertionError(f"mypy produced no output (exit code {returncode}).")

    current_errors = _extract_error_count(output)
    if current_errors <= max_errors:
        return

    offenders = _top_offenders(output)
    offenders_text = "\n".join(f"- {path}: {count}" for path, count in offenders) or "(no offenders parsed)"

    raise AssertionError(
        "mypy error budget exceeded.\n"
        f"Current errors: {current_errors}\n"
        f"Budget: {max_errors}\n"
        "Top offenders:\n"
        f"{offenders_text}\n"
        "Reduce errors or update the budget only when intentional."
    )

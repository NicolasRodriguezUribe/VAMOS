from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], label: str) -> int:
    print(f"\n==> {label}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit {result.returncode})")
        return result.returncode
    print(f"OK: {label}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VAMOS health checks (fast-fail suite).")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run all checks even if one fails.",
    )
    parser.add_argument(
        "--mypy-full",
        action="store_true",
        help="Run full mypy and fail on any errors (stricter than CI).",
    )
    args = parser.parse_args()

    python = sys.executable
    commands: list[tuple[str, list[str]]] = [
        ("Layer boundaries", [python, "-m", "pytest", "-q", "tests/architecture/test_layer_boundaries.py"]),
        ("Monolith guard", [python, "-m", "pytest", "-q", "tests/test_monolith_guard.py"]),
        ("Public API guard", [python, "-m", "pytest", "-q", "tests/test_public_api_guard.py"]),
        ("Import-time smoke", [python, "-m", "pytest", "-q", "tests/test_import_time_smoke.py"]),
        ("Import-time purity", [python, "-m", "pytest", "-q", "tests/architecture/test_no_import_time_side_effects.py"]),
        ("Public API snapshot", [python, "-m", "pytest", "-q", "tests/architecture/test_public_api_snapshot.py"]),
        ("Dependency policy", [python, "-m", "pytest", "-q", "tests/architecture/test_dependency_policy.py"]),
        ("Optional deps policy", [python, "-m", "pytest", "-q", "tests/test_optional_deps_policy.py"]),
        ("Logging policy", [python, "-m", "pytest", "-q", "tests/test_logging_policy.py"]),
        ("No prints in library", [python, "-m", "pytest", "-q", "tests/test_no_prints_in_library.py"]),
        ("No deprecation shims", [python, "-m", "pytest", "-q", "tests/test_no_deprecation_shims.py"]),
        ("AGENTS health link", [python, "-m", "pytest", "-q", "tests/test_agents_health_link.py"]),
        ("Report retention policy", [python, "-m", "pytest", "-q", "tests/architecture/test_report_retention_policy.py"]),
        ("Ruff lint gate", [python, "-m", "pytest", "-q", "tests/architecture/test_ruff_gate.py"]),
        ("Ruff format gate", [python, "-m", "pytest", "-q", "tests/architecture/test_ruff_format_gate.py"]),
        ("Mypy error budget", [python, "-m", "pytest", "-q", "tests/architecture/test_mypy_error_budget.py"]),
        ("Build smoke", [python, "-m", "pytest", "-q", "tests/architecture/test_build_smoke.py"]),
        ("py.typed present", [python, "-m", "pytest", "-q", "tests/architecture/test_py_typed_present.py"]),
        ("Ruff check", [python, "-m", "ruff", "check", "src/vamos", "tests"]),
        ("Ruff format", [python, "-m", "ruff", "format", "--check", "src/vamos", "tests"]),
        ("Build", [python, "-m", "build"]),
    ]
    if args.mypy_full:
        commands.append(("Mypy (full)", [python, "-m", "mypy", "--config-file", "pyproject.toml", "src/vamos"]))

    failures = 0
    for label, cmd in commands:
        code = _run(cmd, label)
        if code != 0:
            failures += 1
            if not args.continue_on_failure:
                return code

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        return 1
    print("\nAll health checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

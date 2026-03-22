from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_compare_pymoo_help() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/benchmark_compare_pymoo.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.backends
def test_benchmark_compare_pymoo_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("pymoo") is None:
        pytest.skip("pymoo not installed")
    if importlib.util.find_spec("moocore") is None:
        pytest.skip("moocore not installed")

    json_path = tmp_path / "comparison.json"
    md_path = tmp_path / "comparison.md"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_compare_pymoo.py",
            "--smoke",
            "--cases",
            "zdt1",
            "--algorithms",
            "nsgaii",
            "--seeds",
            "42",
            "--engine",
            "numpy",
            "--output",
            str(json_path),
            "--markdown",
            str(md_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]
    assert {row["framework"] for row in payload["summary"]} == {"vamos", "pymoo"}
    assert md_path.is_file()

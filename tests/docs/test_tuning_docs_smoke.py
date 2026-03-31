from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_vamos(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.smoke
def test_tuning_docs_smoke_command(tmp_path: Path) -> None:
    source_path = "docs/topics/tuning.md"
    output_root = tmp_path / "tuning_docs"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        "random",
        "--smoke",
        "--output-dir",
        str(output_root),
        "--name",
        "docs_tuning_smoke",
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    summary_path = output_root / "docs_tuning_smoke" / "tuning_summary.json"
    assert summary_path.exists(), source_path

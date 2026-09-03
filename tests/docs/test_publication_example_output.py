from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _status() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def test_publication_ready_default_output_isolated(tmp_path: Path) -> None:
    before = _status()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "advanced" / "publication_ready.py"), "--max-evaluations", "200"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = tmp_path / "artifacts" / "publication" / "zdt1_table.tex"
    assert output.is_file()
    assert "tab:zdt1_nsgaii" in output.read_text(encoding="utf-8")
    assert not (ROOT / "zdt1_table.tex").exists()
    assert _status() == before


def test_plot_example_default_output_isolated(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    before = _status()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["VAMOS_ADV_ENGINE"] = "numpy"
    environment["VAMOS_ADV_EVALS"] = "200"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "advanced" / "plot_nsgaii_variants.py")],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = tmp_path / "results" / "examples" / "nsgaii_variants_dtlz2.png"
    assert output.is_file()
    assert not (ROOT / "nsgaii_variants_dtlz2.png").exists()
    assert _status() == before


def test_publication_ready_refuses_output_collision(tmp_path: Path) -> None:
    output = tmp_path / "existing.tex"
    output.write_text("keep me\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "advanced" / "publication_ready.py"),
            "--output",
            str(output),
            "--max-evaluations",
            "1",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert completed.returncode != 0
    assert "Refusing to overwrite" in completed.stderr
    assert output.read_text(encoding="utf-8") == "keep me\n"

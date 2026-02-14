"""Tests for the consolidated CLI subcommand dispatch.

Verifies that all legacy vamos-* standalone commands are reachable
via `vamos <subcommand>` (e.g. `vamos check`, `vamos bench`).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import pytest

from vamos.engine.tuning import available_model_based_backends


def _run_vamos(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run `python -m vamos.experiment.cli.main <args>` as a subprocess."""
    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


# ---- help ----


def test_help_subcommand_lists_all():
    proc = _run_vamos("help")
    assert proc.returncode == 0
    stdout = proc.stdout.decode()
    # Each consolidated command should appear in help output
    for name in ("quickstart", "create-problem", "summarize", "check", "bench", "studio", "zoo", "tune", "profile"):
        assert name in stdout, f"'{name}' not found in help output"


def test_help_commands_alias():
    proc = _run_vamos("--help-commands")
    assert proc.returncode == 0
    assert "check" in proc.stdout.decode()


# ---- consolidated commands: import smoke ----


def test_check_dispatches():
    """vamos check should call self_check and exit 0."""
    proc = _run_vamos("check")
    assert proc.returncode == 0, proc.stderr.decode()


def test_bench_help():
    """vamos bench --help should work."""
    proc = _run_vamos("bench", "--help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_zoo_help():
    proc = _run_vamos("zoo", "--help")
    assert proc.returncode == 0


# ---- _SUBCOMMANDS dict ----


def test_subcommands_dict_matches_dispatch():
    """The _SUBCOMMANDS constant must list every command that _dispatch_subcommand handles."""
    from vamos.experiment.cli.main import _SUBCOMMANDS

    assert isinstance(_SUBCOMMANDS, dict)
    # At minimum, original + consolidated = 12 entries
    assert len(_SUBCOMMANDS) >= 12
    for name in ("quickstart", "create-problem", "summarize", "check", "bench", "studio", "zoo", "tune", "profile"):
        assert name in _SUBCOMMANDS, f"'{name}' missing from _SUBCOMMANDS"


def test_tune_smoke_with_suite_split(tmp_path):
    out_root = tmp_path / "tune_results"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        "random",
        "--split-strategy",
        "suite_stratified",
        "--budget",
        "30",
        "--tune-budget",
        "4",
        "--n-seeds",
        "2",
        "--n-jobs",
        "1",
        "--no-run-validation",
        "--no-run-test",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(out_root),
        "--name",
        "cli_smoke_tune",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = out_root / "cli_smoke_tune"
    split_path = run_dir / "split_instances.csv"
    summary_path = run_dir / "tuning_summary.json"
    assert split_path.exists()
    assert summary_path.exists()

    with split_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert rows
    assert "suite" in rows[0]
    assert any(row.get("split") == "train" for row in rows)
    assert any(row.get("split") == "validation" for row in rows)
    assert any(row.get("split") == "test" for row in rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("backend_requested") == "random"
    assert summary.get("backend_effective") == "random"
    split_info = summary.get("split", {})
    assert split_info.get("split_strategy") == "suite_stratified"


def test_tune_backend_fallback_subprocess(tmp_path):
    availability = available_model_based_backends()
    unavailable = [name for name, ok in availability.items() if not bool(ok)]
    if not unavailable:
        pytest.skip("All model-based backends are available in this environment.")

    requested = unavailable[0]
    out_root = tmp_path / "tune_results"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        requested,
        "--backend-fallback",
        "random",
        "--split-strategy",
        "suite_stratified",
        "--budget",
        "30",
        "--tune-budget",
        "4",
        "--n-seeds",
        "2",
        "--n-jobs",
        "1",
        "--no-run-validation",
        "--no-run-test",
        "--no-run-statistical-finisher",
        "--output-dir",
        str(out_root),
        "--name",
        "cli_fallback_tune",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    run_dir = out_root / "cli_fallback_tune"
    summary_path = run_dir / "tuning_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("backend_requested") == requested
    assert summary.get("backend_effective") == "random"

"""Tests for the consolidated CLI subcommand dispatch.

Verifies that all legacy vamos-* standalone commands are reachable
via `vamos <subcommand>` (e.g. `vamos check`, `vamos bench`).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


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

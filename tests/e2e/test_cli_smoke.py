import subprocess
import sys

import pytest


@pytest.mark.e2e
def test_cli_help_smoke():
    """
    Ensure the CLI entrypoint is importable and runs.
    """
    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--help"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()

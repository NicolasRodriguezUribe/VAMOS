import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def e2e_workspace():
    """
    Creates a temporary directory for E2E tests and tears it down afterwards.
    Returns the Path to the workspace.
    """
    temp_dir = tempfile.mkdtemp(prefix="vamos_e2e_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def workspace(e2e_workspace):
    return e2e_workspace

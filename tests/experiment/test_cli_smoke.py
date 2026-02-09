import subprocess
import sys

import pytest


def _run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, timeout=30)


def test_self_check_module_smoke():
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.diagnostics.self_check")
    assert proc.returncode == 0, proc.stderr.decode()


def test_vamos_benchmark_help():
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.benchmark.cli --help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_vamos_zoo_help():
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.zoo.cli --help")
    assert proc.returncode == 0


@pytest.mark.studio
def test_vamos_studio_help():
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit not installed")
    proc = _run_cmd(f"{sys.executable} -m vamos.ux.studio.app --help")
    assert proc.returncode == 0

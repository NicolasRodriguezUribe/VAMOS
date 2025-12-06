import subprocess
import sys

import pytest


def _run_cmd(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


def test_self_check_module_smoke():
    proc = _run_cmd(f"{sys.executable} -m vamos.self_check")
    assert proc.returncode == 0, proc.stderr.decode()


def test_vamos_benchmark_help():
    proc = _run_cmd("vamos-benchmark --help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_vamos_zoo_help():
    proc = _run_cmd("vamos-zoo --help")
    assert proc.returncode == 0


@pytest.mark.studio
def test_vamos_studio_help():
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit not installed")
    proc = _run_cmd("vamos-studio --help")
    assert proc.returncode == 0

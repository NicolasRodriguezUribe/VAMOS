import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebooks


def _iter_notebooks():
    nb_dir = Path("notebooks")
    if not nb_dir.exists():
        return []
    return sorted(nb_dir.glob("0*_*.ipynb"))


@pytest.mark.examples
def test_notebooks_execute_smoke(tmp_path):
    if not os.environ.get("VAMOS_RUN_NOTEBOOK_TESTS"):
        pytest.skip("Set VAMOS_RUN_NOTEBOOK_TESTS=1 to execute notebook smoke tests.")
    nbformat = pytest.importorskip("nbformat")
    nbconvert = pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor

    for nb_path in _iter_notebooks():
        with nb_path.open("r", encoding="utf-8") as fh:
            nb = nbformat.read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": tmp_path}})

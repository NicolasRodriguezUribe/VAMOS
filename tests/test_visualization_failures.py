import sys

import pytest


def test_plot_functions_require_matplotlib(monkeypatch):
    # Remove matplotlib from sys.modules and block import to simulate minimal install
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    from vamos import visualization

    with pytest.raises(ImportError):
        visualization.plot_pareto_front_2d([[0.0, 1.0], [1.0, 0.0]])

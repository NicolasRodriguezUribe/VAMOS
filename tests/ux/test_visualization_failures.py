import sys

import pytest


def test_plot_functions_require_matplotlib(monkeypatch):
    # Remove matplotlib from sys.modules and block import to simulate minimal install
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    import vamos.plotting as plotting

    with pytest.raises(ImportError):
        plotting.plot_pareto_front_2d([[0.0, 1.0], [1.0, 0.0]])

import pytest
import numpy as np
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vamos.ux.visualization import (
    plot_hv_convergence,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)


def test_plot_pareto_front_2d_smoke():
    F = np.array([[1.0, 2.0], [0.5, 1.5]])
    ax = plot_pareto_front_2d(F, show=False)
    assert hasattr(ax, "scatter")


def test_plot_pareto_front_3d_smoke():
    F = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    ax = plot_pareto_front_3d(F, show=False)
    assert hasattr(ax, "scatter")


def test_plot_parallel_coordinates_smoke():
    F = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    ax = plot_parallel_coordinates(F, show=False)
    assert hasattr(ax, "plot")


def test_plot_hv_convergence_smoke():
    evals = np.array([0, 10, 20])
    hv = np.array([0.1, 0.2, 0.3])
    ax = plot_hv_convergence(evals, hv, show=False)
    assert hasattr(ax, "plot")

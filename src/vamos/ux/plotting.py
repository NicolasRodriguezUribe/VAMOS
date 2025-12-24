"""
Plotting helpers exposed for end users.

Thin wrappers around `vamos.ux.visualization` for quick Pareto/front plots.
"""
from __future__ import annotations

from vamos.ux.visualization import (
    plot_hv_convergence,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)

__all__ = [
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    "plot_hv_convergence",
]


from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.problem.types import ProblemProtocol
from vamos.ux.visualization import plot_pareto_front_2d, plot_pareto_front_3d


def plot_quick_front(
    *,
    F: np.ndarray,
    algorithm: str,
    problem: ProblemProtocol,
    show: bool = True,
    title: str | None = None,
    labels: tuple[str, str] | tuple[str, str, str] | None = None,
) -> Any:
    """Plot a 2D or 3D Pareto front for a quick run."""
    n_obj = F.shape[1] if F.ndim == 2 else 1
    default_title = f"{algorithm.upper()} on {type(problem).__name__}"

    if n_obj == 2:
        return plot_pareto_front_2d(
            F,
            title=title or default_title,
            labels=labels,
            show=show,
        )
    if n_obj == 3:
        return plot_pareto_front_3d(
            F,
            title=title or default_title,
            labels=labels,
            show=show,
        )
    raise ValueError(
        f"Cannot plot {n_obj}-objective front directly. "
        "Use parallel coordinates or reduce objectives first."
    )

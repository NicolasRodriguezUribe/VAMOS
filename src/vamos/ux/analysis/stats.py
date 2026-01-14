from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

import numpy as np
from scipy import stats as spstats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "Statistics plotting requires matplotlib. Install with `pip install vamos[analysis]` or `pip install matplotlib`."
        ) from exc


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    ranks: np.ndarray
    avg_ranks: np.ndarray


@dataclass
class WilcoxonResult:
    algo_i: str
    algo_j: str
    statistic: float
    p_value: float


def compute_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Compute per-problem ranks for algorithms.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    n_problems, n_algos = scores.shape
    ranks = np.empty_like(scores)
    for p in range(n_problems):
        vals = scores[p]
        if not higher_is_better:
            vals = -vals
        ranks[p] = spstats.rankdata(-vals, method="average")  # -vals so best gets rank 1
    return ranks


def friedman_test(scores: np.ndarray, higher_is_better: bool = True) -> FriedmanResult:
    """
    Run Friedman test using ranks across problems.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    ranks = compute_ranks(scores, higher_is_better=higher_is_better)
    avg_ranks = np.mean(ranks, axis=0)
    # scipy expects columns; we run on raw scores as in classic use
    stat, p = spstats.friedmanchisquare(*[scores[:, j] if higher_is_better else -scores[:, j] for j in range(scores.shape[1])])
    return FriedmanResult(statistic=stat, p_value=p, ranks=ranks, avg_ranks=avg_ranks)


def pairwise_wilcoxon(
    scores: np.ndarray,
    algo_names: Sequence[str],
    higher_is_better: bool = True,
    alpha: float = 0.05,
) -> list[WilcoxonResult]:
    """
    Pairwise Wilcoxon signed-rank tests across algorithms.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be 2-dimensional (n_problems, n_algorithms).")
    n_algos = scores.shape[1]
    if len(algo_names) != n_algos:
        raise ValueError("algo_names length must match number of algorithms.")
    results: list[WilcoxonResult] = []
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            a = scores[:, i]
            b = scores[:, j]
            if not higher_is_better:
                a = -a
                b = -b
            stat, p = spstats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
            results.append(WilcoxonResult(algo_i=algo_names[i], algo_j=algo_names[j], statistic=stat, p_value=p))
    return results


def plot_critical_distance(
    avg_ranks: np.ndarray,
    algo_names: Sequence[str],
    alpha: float = 0.05,
    n_problems: int | None = None,
    higher_is_better: bool = True,
    ax: Axes | None = None,
    show: bool = False,
) -> object:
    """
    Plot a simple critical distance diagram using Nemenyi CD if n_problems provided.
    """
    plt = _require_matplotlib()
    avg_ranks = np.asarray(avg_ranks, dtype=float)
    if avg_ranks.ndim != 1:
        raise ValueError("avg_ranks must be 1-dimensional.")
    if len(algo_names) != avg_ranks.shape[0]:
        raise ValueError("algo_names length must match avg_ranks length.")
    k = avg_ranks.shape[0]
    ax = ax or plt.figure().add_subplot(111)
    ranks = avg_ranks if higher_is_better else (k + 1 - avg_ranks)
    y = np.zeros_like(ranks)
    ax.scatter(ranks, y, c="black")
    for idx, name in enumerate(algo_names):
        ax.text(ranks[idx], 0.02, name, rotation=45, ha="center", va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("Average Rank (lower is better)")
    if n_problems is not None:
        # q_alpha approximation for alpha=0.05 (large-sample Studentized range)
        q_alpha = 2.569 if alpha == 0.05 else 2.343  # simple fallback
        cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_problems))
        xmin = ranks.min()
        ax.hlines(0.1, xmin, xmin + cd, colors="black")
        ax.vlines([xmin, xmin + cd], 0.08, 0.12, colors="black")
        ax.text(xmin + cd / 2, 0.13, f"CD={cd:.2f}", ha="center")
    if show:
        plt.show()
    return ax

"""
Plot helpers for benchmark reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional heavy dep
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

from vamos.ux.analysis.stats import plot_critical_distance


def generate_plots(
    tidy, stats: Dict[str, dict], metrics: List[str], alpha: float, suite_name: str, output_dir: Path, higher_is_better
) -> Dict[str, List[Path]]:
    if plt is None:
        return {}
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, List[Path]] = {}
    for metric in metrics:
        metric_paths: List[Path] = []
        payload = stats.get(metric, {})
        fried = payload.get("friedman")
        if fried and getattr(fried, "p_value", 1.0) is not None and fried.p_value < alpha:
            pivot_mean = payload.get("pivot_mean")
            if pivot_mean is not None:
                fig, ax = plt.subplots()
                plot_critical_distance(
                    avg_ranks=fried.avg_ranks,
                    algo_names=list(pivot_mean.columns),
                    alpha=alpha,
                    n_problems=pivot_mean.shape[0],
                    higher_is_better=higher_is_better(metric),
                    ax=ax,
                    show=False,
                )
                cd_path = plots_dir / f"cd_{metric}.pdf"
                fig.tight_layout()
                fig.savefig(cd_path)
                plt.close(fig)
                metric_paths.append(cd_path)
        dfm = tidy[tidy["metric"] == metric]
        problems = sorted(dfm["problem"].unique())
        for problem in problems:
            subset = dfm[dfm["problem"] == problem]
            fig, ax = plt.subplots()
            data = [subset[subset["algorithm"] == alg]["value"].dropna().values for alg in subset["algorithm"].unique()]
            labels = list(subset["algorithm"].unique())
            ax.boxplot(data, tick_labels=labels, patch_artist=True)
            ax.set_title(f"{metric} | {problem}")
            ax.set_ylabel(metric)
            fig.tight_layout()
            path = plots_dir / f"boxplot_{metric}_{problem}.pdf"
            fig.savefig(path)
            plt.close(fig)
            metric_paths.append(path)
        if metric_paths:
            created[metric] = metric_paths
    return created


__all__ = ["generate_plots"]

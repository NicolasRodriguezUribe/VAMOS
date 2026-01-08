from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from vamos.experiment.benchmark.report_utils import ensure_dir, import_pandas


def _load_matplotlib():
    try:  # pragma: no cover - optional heavy dep
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None, None
    return matplotlib, plt


def generate_boxplots(summary, output_dir: Path) -> Dict[str, List[Path]]:
    pd = import_pandas()
    _, plt = _load_matplotlib()
    if plt is None:
        return {}
    output_dir = ensure_dir(output_dir)
    created: Dict[str, List[Path]] = {}

    algorithms = pd.unique(summary["Algorithm"])
    problems = pd.unique(summary["Problem"])
    indicators = pd.unique(summary["IndicatorName"])

    for indicator_name in indicators:
        data = summary[summary["IndicatorName"] == indicator_name]
        paths: List[Path] = []
        for problem in problems:
            data_to_plot = []
            for alg in algorithms:
                subset = data[(data["Algorithm"] == alg) & (data["Problem"] == problem)]
                data_to_plot.append(subset["IndicatorValue"].values)

            fig = plt.figure(1, figsize=(9, 6))
            plt.suptitle(problem, y=0.95, fontsize=14)
            ax = fig.add_subplot(111)
            ax.boxplot(data_to_plot)
            ax.set_xticklabels(algorithms, rotation=45, ha="right")
            ax.tick_params(labelsize=10)

            png_path = output_dir / f"boxplot-{problem}-{indicator_name}.png"
            pdf_path = output_dir / f"boxplot-{problem}-{indicator_name}.pdf"
            fig.savefig(png_path, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            paths.extend([png_path, pdf_path])
        if paths:
            created[indicator_name] = paths
    return created


__all__ = ["generate_boxplots"]

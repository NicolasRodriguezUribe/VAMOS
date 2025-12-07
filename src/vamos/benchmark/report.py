from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover - optional heavy dep
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - allow headless environments
    matplotlib = None
    plt = None

from vamos.benchmark.runner import BenchmarkResult
from vamos.stats import friedman_test, pairwise_wilcoxon, plot_critical_distance


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Benchmark reporting requires pandas. Install via 'pip install pandas' or the "
            "'notebooks'/'examples' extras."
        ) from exc
    return pd


@dataclass
class BenchmarkReportConfig:
    metrics: List[str]
    alpha: float = 0.05
    latex_float_format: str = "%.3f"
    table_caption_prefix: str = ""
    table_label_prefix: str = ""


class BenchmarkReport:
    def __init__(self, result: BenchmarkResult, config: BenchmarkReportConfig, output_dir: Path):
        self.result = result
        self.config = config
        self.output_dir = _ensure_dir(output_dir)
        self._tidy = None
        self._stats_cache: Dict[str, Any] | None = None

    @staticmethod
    def _higher_is_better(metric: str) -> bool:
        m = metric.lower()
        if m in {"igd", "igd+", "igd_plus", "epsilon", "epsilon_additive", "epsilon_mult"}:
            return False
        return True

    def aggregate_metrics(self):
        if self._tidy is not None:
            return self._tidy
        pd = _import_pandas()
        summary_path = self.result.summary_path
        if summary_path is None or not summary_path.exists():
            raise FileNotFoundError("Summary CSV not found for benchmark reporting.")
        raw = pd.read_csv(summary_path)
        records: list[dict[str, Any]] = []
        for _, row in raw.iterrows():
            for metric in self.config.metrics:
                col = metric if metric in row else None
                if col is None and f"indicator_{metric}" in row:
                    col = f"indicator_{metric}"
                if col is None:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                records.append(
                    {
                        "suite": self.result.suite.name,
                        "problem": row.get("problem"),
                        "algorithm": row.get("algorithm"),
                        "engine": row.get("engine"),
                        "seed": row.get("seed"),
                        "metric": metric,
                        "value": val,
                        "n_var": row.get("n_var"),
                        "n_obj": row.get("n_obj"),
                    }
                )
        tidy = pd.DataFrame.from_records(records)
        self._tidy = tidy
        tidy_path = self.output_dir / "metrics_tidy.csv"
        tidy.to_csv(tidy_path, index=False)
        return tidy

    def compute_statistics(self) -> Dict[str, Any]:
        if self._stats_cache is not None:
            return self._stats_cache
        pd = _import_pandas()
        tidy = self.aggregate_metrics()
        stats: Dict[str, Any] = {}
        for metric in self.config.metrics:
            dfm = tidy[tidy["metric"] == metric]
            if dfm.empty:
                continue
            grouped = dfm.groupby(["problem", "algorithm"])["value"].agg(["mean", "std"])
            pivot_mean = grouped["mean"].unstack("algorithm")
            fried = None
            wilc = []
            if pivot_mean.shape[1] > 1 and pivot_mean.shape[0] > 1:
                scores = pivot_mean.values
                # Friedman requires at least 3 algorithms; skip when not applicable.
                if scores.shape[1] >= 3:
                    fried = friedman_test(scores, higher_is_better=self._higher_is_better(metric))
                    wilc = pairwise_wilcoxon(
                        scores,
                        list(pivot_mean.columns),
                        higher_is_better=self._higher_is_better(metric),
                        alpha=self.config.alpha,
                    )
            stats[metric] = {"grouped": grouped, "pivot_mean": pivot_mean, "friedman": fried, "wilcoxon": wilc}
        self._stats_cache = stats
        # Persist quick summary
        summary_path = self.output_dir / "statistics.json"
        serializable = {}
        for metric, payload in stats.items():
            fried = payload.get("friedman")
            serializable[metric] = {
                "friedman": {
                    "statistic": getattr(fried, "statistic", None),
                    "p_value": getattr(fried, "p_value", None),
                }
                if fried
                else None
            }
        summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return stats

    def _best_algorithms(self, mean_table, metric: str) -> Dict[str, str]:
        best: Dict[str, str] = {}
        higher = self._higher_is_better(metric)
        for problem in mean_table.index:
            row = mean_table.loc[problem]
            if higher:
                best_alg = row.idxmax()
            else:
                best_alg = row.idxmin()
            best[problem] = best_alg
        return best

    def _format_cell(self, mean: float, std: float, is_best: bool, marker: str) -> str:
        fmt = self.config.latex_float_format
        if fmt.startswith("%"):
            fmt = fmt.lstrip("%")
        cell = f"{mean:{fmt}} +/- {std:{fmt}}"
        if is_best:
            cell = f"\\textbf{{{cell}}}"
        if marker:
            cell = f"{cell}$^{{{marker}}}$"
        return cell

    def _per_problem_markers(self, dfm, problem: str, best_alg: str, metric: str) -> Dict[str, str]:
        markers: Dict[str, str] = {}
        try:
            from scipy import stats as spstats  # type: ignore
        except Exception:  # pragma: no cover - optional
            return markers
        subset = dfm[dfm["problem"] == problem]
        pivot = subset.pivot_table(index="seed", columns="algorithm", values="value")
        if best_alg not in pivot.columns:
            return markers
        best_values = pivot[best_alg].dropna()
        for alg in pivot.columns:
            if alg == best_alg:
                continue
            other = pivot[alg].dropna()
            if best_values.empty or other.empty:
                continue
            try:
                stat, p = spstats.wilcoxon(best_values, other, zero_method="pratt", alternative="two-sided")
            except ValueError:
                continue
            if p < self.config.alpha:
                higher = self._higher_is_better(metric)
                best_mean = best_values.mean()
                other_mean = other.mean()
                if (higher and other_mean > best_mean) or ((not higher) and other_mean < best_mean):
                    markers[alg] = "+"
                else:
                    markers[alg] = "-"
        return markers

    def generate_latex_tables(self) -> Dict[str, Path]:
        pd = _import_pandas()
        tidy = self.aggregate_metrics()
        stats = self.compute_statistics()
        tables_dir = _ensure_dir(self.output_dir / "tables")
        created: Dict[str, Path] = {}
        for metric in self.config.metrics:
            dfm = tidy[tidy["metric"] == metric]
            if dfm.empty:
                continue
            grouped = dfm.groupby(["problem", "algorithm"])["value"].agg(["mean", "std"])
            mean_table = grouped["mean"].unstack("algorithm")
            std_table = grouped["std"].unstack("algorithm")
            best_map = self._best_algorithms(mean_table, metric)
            rows = []
            problems = mean_table.index.tolist()
            algorithms = mean_table.columns.tolist()
            for problem in problems:
                row_cells = {"Problem": problem}
                markers = self._per_problem_markers(dfm, problem, best_map[problem], metric)
                for alg in algorithms:
                    mean = mean_table.loc[problem, alg]
                    std = std_table.loc[problem, alg]
                    is_best = alg == best_map[problem]
                    marker = markers.get(alg, "")
                    row_cells[alg] = self._format_cell(mean, std, is_best, marker)
                rows.append(row_cells)
            table_df = pd.DataFrame(rows)
            latex_path = tables_dir / f"{self.result.suite.name}_{metric}.tex"
            caption = f"{self.config.table_caption_prefix}{metric} results for {self.result.suite.name}"
            label = f"{self.config.table_label_prefix}{self.result.suite.name}_{metric}"
            table_df.to_latex(
                latex_path,
                index=False,
                escape=False,
                caption=caption,
                label=label,
            )
            created[metric] = latex_path
        if created:
            master = tables_dir / "all_tables.tex"
            master.write_text(
                "\n".join(f"\\input{{{path.name}}}" for path in created.values()),
                encoding="utf-8",
            )
            created["all"] = master
        return created

    def generate_plots(self) -> Dict[str, List[Path]]:
        if plt is None:
            return {}
        tidy = self.aggregate_metrics()
        stats = self.compute_statistics()
        plots_dir = _ensure_dir(self.output_dir / "plots")
        created: Dict[str, List[Path]] = {}
        for metric in self.config.metrics:
            metric_paths: List[Path] = []
            payload = stats.get(metric, {})
            fried = payload.get("friedman")
            if fried and getattr(fried, "p_value", 1.0) is not None and fried.p_value < self.config.alpha:
                pivot_mean = payload.get("pivot_mean")
                if pivot_mean is not None:
                    fig, ax = plt.subplots()
                    plot_critical_distance(
                        avg_ranks=fried.avg_ranks,
                        algo_names=list(pivot_mean.columns),
                        alpha=self.config.alpha,
                        n_problems=pivot_mean.shape[0],
                        higher_is_better=self._higher_is_better(metric),
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
                ax.boxplot(data, labels=labels, patch_artist=True)
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




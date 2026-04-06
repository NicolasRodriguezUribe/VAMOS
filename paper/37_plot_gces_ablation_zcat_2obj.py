"""Generate diagnostic figures for the GCES 2-objective ZCAT ablation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gces_ablation_zcat_2obj_common import ALGORITHM_LABELS, DEFAULT_OUTPUT_DIR, DEFAULT_PROBLEMS, load_run_config


DELTA_ALGORITHMS = ["gces_nocomp", "gces_nogeo", "gces"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _plot_heatmaps(output_dir: Path, comparison_rows: list[dict[str, str]], problems: list[str]) -> None:
    metrics = [
        ("hypervolume", "Median HV delta vs nsgaii (higher is better)"),
        ("igd_plus", "Median IGD+ delta vs nsgaii (negative is better)"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 10), constrained_layout=True)

    for axis, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.zeros((len(problems), len(DELTA_ALGORITHMS)), dtype=float)
        for row_idx, problem in enumerate(problems):
            for col_idx, algorithm in enumerate(DELTA_ALGORITHMS):
                row = next(
                    item
                    for item in comparison_rows
                    if item["problem"] == problem and item["metric"] == metric and item["lhs_algorithm"] == algorithm and item["rhs_algorithm"] == "nsgaii"
                )
                matrix[row_idx, col_idx] = float(row["delta_median"])

        vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        vmax = max(vmax, 1e-12)
        im = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(DELTA_ALGORITHMS)), [ALGORITHM_LABELS[algorithm] for algorithm in DELTA_ALGORITHMS], rotation=20, ha="right")
        axis.set_yticks(range(len(problems)), problems)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:+.3f}", ha="center", va="center", fontsize=7)
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    figure.savefig(output_dir / "median_delta_heatmaps.png", dpi=200)
    plt.close(figure)


def _plot_hv_boxplots(output_dir: Path, raw_rows: list[dict[str, str]], problems: list[str], algorithms: list[str]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["problem"], row["algorithm"])].append(float(row["hypervolume"]))

    figure, axes = plt.subplots(5, 4, figsize=(16, 14), sharey=False, constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, problem in zip(flat_axes, problems, strict=False):
        data = [grouped[(problem, algorithm)] for algorithm in algorithms]
        axis.boxplot(data, tick_labels=[ALGORITHM_LABELS[algorithm] for algorithm in algorithms], showfliers=True)
        axis.set_title(problem)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("HV")
    for axis in flat_axes[len(problems) :]:
        axis.axis("off")

    figure.suptitle("Per-seed HV distributions by problem and algorithm", fontsize=14)
    figure.savefig(output_dir / "hv_boxplots.png", dpi=200)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate diagnostic figures for the GCES ablation campaign.")
    parser.add_argument("--input-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing raw_results.csv, comparison.csv, and run_config.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(input_dir / "run_config.json")
    comparison_rows = _read_csv(input_dir / "comparison.csv")
    raw_rows = _read_csv(input_dir / "raw_results.csv")
    problems = list(run_config.get("problems", DEFAULT_PROBLEMS))
    algorithms = list(run_config.get("algorithms", list(ALGORITHM_LABELS.keys())))

    _plot_heatmaps(output_dir, comparison_rows, problems)
    _plot_hv_boxplots(output_dir, raw_rows, problems, algorithms)
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

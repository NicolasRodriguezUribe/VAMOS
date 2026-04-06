"""Generate plots for the 3-objective ZCAT difficulty robustness campaign."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from farthest_gces_difficulty_ab_zcat_3obj_common import ALGORITHM_LABELS
from gces_ablation_zcat_2obj_common import load_run_config

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "farthest_gces_difficulty_ab_zcat_3obj_full"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _plot_global_trends(output_dir: Path, global_rows: list[dict[str, str]], algorithms: list[str], labels: dict[str, str], config_order: list[str], config_labels: dict[str, str]) -> None:
    metrics = [
        ("hv_median_of_medians", "HV median-of-medians", True),
        ("igd_plus_median_of_medians", "IGD+ median-of-medians", False),
        ("runtime_seconds_median_of_medians", "Runtime median-of-medians (s)", False),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    x = np.arange(len(config_order))
    for axis, (field, title, higher_is_better) in zip(axes, metrics, strict=True):
        for algorithm in algorithms:
            y = [
                float(
                    next(row for row in global_rows if row["config_id"] == config_id and row["algorithm"] == algorithm)[field]
                )
                for config_id in config_order
            ]
            axis.plot(x, y, marker="o", label=labels[algorithm])
        axis.set_title(title)
        axis.set_xticks(x, [config_labels[config_id] for config_id in config_order], rotation=20)
        axis.grid(alpha=0.25)
        if not higher_is_better:
            axis.set_ylim(bottom=0.0)
    axes[0].legend(loc="best", fontsize=8)
    figure.savefig(output_dir / "difficulty_global_trends.png", dpi=200)
    plt.close(figure)


def _plot_advantage_heatmaps(output_dir: Path, global_rows: list[dict[str, str]], algorithms: list[str], labels: dict[str, str], config_order: list[str], config_labels: dict[str, str]) -> None:
    compared = [algorithm for algorithm in algorithms if algorithm != "nsgaii"]
    panels = [
        ("hv_delta_vs_nsgaii", "HV delta vs nsgaii"),
        ("igd_plus_improvement_vs_nsgaii", "IGD+ improvement vs nsgaii"),
        ("runtime_ratio_vs_nsgaii", "Runtime ratio vs nsgaii"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for axis, (field, title) in zip(axes, panels, strict=True):
        matrix = np.array(
            [
                [
                    float(
                        next(row for row in global_rows if row["config_id"] == config_id and row["algorithm"] == algorithm)[field]
                    )
                    for algorithm in compared
                ]
                for config_id in config_order
            ],
            dtype=float,
        )
        if field == "runtime_ratio_vs_nsgaii":
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
            cmap = "viridis"
        else:
            vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
            vmax = max(vmax, 1e-12)
            vmin = -vmax
            cmap = "coolwarm"
        im = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(compared)), [labels[algorithm] for algorithm in compared], rotation=30, ha="right")
        axis.set_yticks(range(len(config_order)), [config_labels[config_id] for config_id in config_order])
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:+.3f}" if field != "runtime_ratio_vs_nsgaii" else f"{matrix[row_idx, col_idx]:.2f}x", ha="center", va="center", fontsize=7)
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(output_dir / "difficulty_advantage_heatmaps.png", dpi=200)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots for the 3-objective ZCAT difficulty robustness campaign."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing campaign CSV outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(input_dir / "run_config.json")
    global_rows = _read_csv(input_dir / "global_summary.csv")
    algorithms = list(run_config["algorithms"])
    labels = {
        algorithm: run_config.get("algorithm_labels", {}).get(algorithm, ALGORITHM_LABELS.get(algorithm, algorithm))
        for algorithm in algorithms
    }
    config_order = [cell["config_id"] for cell in run_config["difficulty_cells"]]
    config_labels = {cell["config_id"]: cell["config_label"] for cell in run_config["difficulty_cells"]}

    _plot_global_trends(output_dir, global_rows, algorithms, labels, config_order, config_labels)
    _plot_advantage_heatmaps(output_dir, global_rows, algorithms, labels, config_order, config_labels)
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

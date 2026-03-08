#!/usr/bin/env python
"""Ablation benchmark for PAVE intensification on ZDT1-3.

This benchmark separates architectural correctness from operator usefulness by
comparing:

1. Legacy NSGA-II baseline: SBX -> repair -> polynomial mutation -> repair
2. New-order control: SBX -> polynomial mutation -> no-op intensification -> repair
3. NSGA-II + PAVE: SBX -> polynomial mutation -> PAVE -> repair
4. NSGA-II + directional: SBX -> polynomial mutation -> directional -> repair

The script uses the normal VAMOS optimization path:
- problems from ``vamos.problems``
- configuration through ``NSGAIIConfig``
- execution via ``vamos.optimize``
- metrics from ``vamos.foundation.quality_indicators``
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.foundation.quality_indicators import (
    IGDPlusIndicator,
    compute_hypervolume,
    get_zdt_reference_front,
    get_zdt_reference_point,
    has_moocore,
)
from vamos.problems import ZDT1, ZDT2, ZDT3

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 200
DEFAULT_SEEDS = 20
DEFAULT_ENGINE = "numpy"
DEFAULT_N_VAR = 30
OUTPUT_PATH = Path(__file__).with_name("results_pave_ablation.json")
PLOT_DIR = Path(__file__).with_name("plots") / "pave_ablation"


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    factory: Callable[..., Any]
    n_var: int = DEFAULT_N_VAR


@dataclass(frozen=True)
class BenchmarkVariant:
    label: str
    crossover: tuple[str, dict[str, Any]]
    intensification: tuple[str, dict[str, Any]] | None = None
    stack: str = ""
    repair_order: str = ""


PROBLEMS = (
    ProblemSpec("zdt1", ZDT1),
    ProblemSpec("zdt2", ZDT2),
    ProblemSpec("zdt3", ZDT3),
)

VARIANTS = (
    BenchmarkVariant(
        label="NSGAII-SBX",
        crossover=("sbx", {"prob": 0.9, "eta": 20.0}),
        stack="SBX -> repair -> polynomial mutation -> repair",
        repair_order="legacy: crossover -> repair -> mutation -> repair",
    ),
    BenchmarkVariant(
        label="NSGAII-SBX+NewOrder",
        crossover=("sbx", {"prob": 0.9, "eta": 20.0}),
        intensification=(
            "directional",
            {
                "prob": 0.0,
                "k_neighbors": 1,
                "alpha": 0.0,
                "beta": 0.0,
            },
        ),
        stack="SBX -> polynomial mutation -> no-op intensification control -> repair",
        repair_order="new-order control: crossover -> mutation -> intensification(no-op) -> repair",
    ),
    BenchmarkVariant(
        label="NSGAII-SBX+PAVE",
        crossover=("sbx", {"prob": 0.9, "eta": 20.0}),
        intensification=(
            "pave",
            {
                "prob": 0.9,
                "k_neighbors": 5,
                "alpha": 0.35,
                "beta": 0.2,
                "lambda_distance": 0.5,
            },
        ),
        stack="SBX -> polynomial mutation -> PAVE intensification -> repair",
        repair_order="intensified: crossover -> mutation -> intensification -> repair",
    ),
    BenchmarkVariant(
        label="NSGAII-SBX+Directional",
        crossover=("sbx", {"prob": 0.9, "eta": 20.0}),
        intensification=(
            "directional",
            {
                "prob": 0.9,
                "k_neighbors": 5,
                "alpha": 0.25,
                "beta": 0.1,
            },
        ),
        stack="SBX -> polynomial mutation -> directional intensification -> repair",
        repair_order="intensified: crossover -> mutation -> intensification -> repair",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds. Runs use range(seeds).")
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true", help="Skip Pareto-front and HV-convergence plots.")
    return parser.parse_args()


def _max_evaluations(population_size: int, generations: int) -> int:
    return population_size * (generations + 1)


def _summary_stats(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None, None
    return float(statistics.fmean(finite)), float(statistics.pstdev(finite))


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def print_variant_stacks() -> None:
    print()
    print("Variant Stacks")
    print("--------------")
    for variant in VARIANTS:
        intensification = "none" if variant.intensification is None else variant.intensification[0]
        print(
            f"- {variant.label}: crossover=sbx, mutation=polynomial, "
            f"intensification={intensification}, stack={variant.stack}"
        )


def _build_generation_callback(ref_point: np.ndarray) -> tuple[Callable[[dict[str, Any]], bool], list[int], list[float]]:
    evals: list[int] = []
    hv_values: list[float] = []

    def _on_generation(payload: dict[str, Any]) -> bool:
        nondominated = payload.get("nondominated")
        population = payload.get("population")

        if isinstance(nondominated, dict) and nondominated.get("F") is not None:
            front_F = np.asarray(nondominated["F"], dtype=float)
        elif isinstance(population, dict) and population.get("F") is not None:
            front_F = np.asarray(population["F"], dtype=float)
        else:
            return False

        if front_F.ndim != 2 or front_F.shape[0] == 0:
            return False

        evaluations = int(payload.get("evaluations") or 0)
        evals.append(evaluations)
        hv_values.append(float(compute_hypervolume(front_F, ref_point)))
        return False

    return _on_generation, evals, hv_values


def build_config(
    *,
    n_var: int,
    population_size: int,
    crossover: tuple[str, dict[str, Any]],
    intensification: tuple[str, dict[str, Any]] | None,
    generation_callback: Callable[[dict[str, Any]], bool] | None,
) -> NSGAIIConfig:
    builder = (
        NSGAIIConfig.builder()
        .pop_size(population_size)
        .offspring_size(population_size)
        .crossover(*crossover)
        .mutation("polynomial", prob=1.0 / n_var, eta=20.0)
        .selection("tournament", size=2)
        .result_mode("population")
    )
    if generation_callback is not None:
        builder = builder.live_callback_mode("population").generation_callback(generation_callback, copy_arrays=False)
    if intensification is not None:
        builder = builder.intensification(*intensification)
    return builder.build()


def run_variant(
    problem_spec: ProblemSpec,
    variant: BenchmarkVariant,
    seeds: list[int],
    *,
    population_size: int,
    generations: int,
    engine: str,
    ref_point: np.ndarray,
    igd_indicator: IGDPlusIndicator | None,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    max_evaluations = _max_evaluations(population_size, generations)
    run_rows: list[dict[str, Any]] = []
    hv_values: list[float | None] = []
    igd_values: list[float | None] = []
    runtime_values: list[float] = []

    for seed in seeds:
        callback, curve_evals, curve_hv = _build_generation_callback(ref_point)
        cfg = build_config(
            n_var=problem_spec.n_var,
            population_size=population_size,
            crossover=variant.crossover,
            intensification=variant.intensification,
            generation_callback=callback,
        )
        problem = problem_spec.factory(n_var=problem_spec.n_var)

        start = time.perf_counter()
        result = optimize(
            problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("max_evaluations", max_evaluations),
            seed=seed,
            engine=engine,
        )
        elapsed = time.perf_counter() - start

        population = result.data.get("population") or {}
        population_X = np.asarray(population.get("X", result.X), dtype=float)
        population_F = np.asarray(population.get("F", result.F), dtype=float)

        front_with_idx = result.front(return_indices=True)
        if front_with_idx is None:
            front_F = population_F
            front_X = population_X
        else:
            front_F, front_idx = front_with_idx
            front_X = population_X[front_idx] if population_X.ndim == 2 else None

        hv = float(compute_hypervolume(front_F, ref_point))
        igd_plus = float(igd_indicator.compute(front_F).value) if igd_indicator is not None else None

        hv_values.append(hv)
        igd_values.append(igd_plus)
        runtime_values.append(elapsed)
        run_rows.append(
            {
                "problem": problem_spec.name,
                "algorithm": variant.label,
                "seed": seed,
                "runtime_seconds": elapsed,
                "evaluations": max_evaluations,
                "population_size": int(population_F.shape[0]),
                "front_size": int(front_F.shape[0]),
                "hv": hv,
                "igd_plus": igd_plus,
                "hv_curve": {
                    "evaluations": curve_evals,
                    "values": curve_hv,
                },
                "population": {
                    "X": population_X,
                    "F": population_F,
                },
                "front": {
                    "X": front_X,
                    "F": front_F,
                },
            }
        )

    hv_mean, hv_std = _summary_stats(hv_values)
    igd_mean, igd_std = _summary_stats(igd_values)
    runtime_mean, runtime_std = _summary_stats(runtime_values)
    return run_rows, {
        "hv_mean": hv_mean,
        "hv_std": hv_std,
        "igd_plus_mean": igd_mean,
        "igd_plus_std": igd_std,
        "runtime_mean_seconds": runtime_mean,
        "runtime_std_seconds": runtime_std,
    }


def _select_representative_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (float(row["hv"]), int(row["seed"])))
    return ordered[len(ordered) // 2]


def _aggregate_mean_hv_curve(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by_eval: dict[int, list[float]] = {}
    for row in rows:
        curve = row.get("hv_curve", {})
        evals = curve.get("evaluations") or []
        values = curve.get("values") or []
        for eval_count, hv in zip(evals, values, strict=False):
            by_eval.setdefault(int(eval_count), []).append(float(hv))
    eval_axis = np.asarray(sorted(by_eval), dtype=float)
    hv_mean = np.asarray([statistics.fmean(by_eval[int(eval_count)]) for eval_count in eval_axis], dtype=float)
    return eval_axis, hv_mean


def _hv_deltas(
    summary: dict[str, dict[str, float | None]],
    runs_by_algorithm: dict[str, list[dict[str, Any]]],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left = {int(row["seed"]): float(row["hv"]) for row in runs_by_algorithm[left_label]}
    right = {int(row["seed"]): float(row["hv"]) for row in runs_by_algorithm[right_label]}
    common_seeds = sorted(set(left) & set(right))
    deltas = [right[seed] - left[seed] for seed in common_seeds]

    left_igd = summary[left_label].get("igd_plus_mean")
    right_igd = summary[right_label].get("igd_plus_mean")
    igd_mean_delta = None
    if left_igd is not None and right_igd is not None:
        igd_mean_delta = float(left_igd) - float(right_igd)

    return {
        "left": left_label,
        "right": right_label,
        "paired_seed_count": len(common_seeds),
        "hv_mean_delta": float(statistics.fmean(deltas)) if deltas else None,
        "hv_better_seed_fraction": (sum(delta > 1.0e-12 for delta in deltas) / len(deltas)) if deltas else None,
        "hv_worse_seed_fraction": (sum(delta < -1.0e-12 for delta in deltas) / len(deltas)) if deltas else None,
        "igd_plus_mean_delta": igd_mean_delta,
    }


def _comparison_text(problem_name: str, comparison: dict[str, Any]) -> str:
    hv_delta = comparison.get("hv_mean_delta")
    better_fraction = comparison.get("hv_better_seed_fraction")
    igd_delta = comparison.get("igd_plus_mean_delta")
    right_label = str(comparison["right"])
    left_label = str(comparison["left"])

    if hv_delta is None or better_fraction is None:
        return f"{problem_name.upper()}: {right_label} vs {left_label} is inconclusive."

    if hv_delta > 1.0e-4:
        verdict = "improves"
    elif hv_delta < -1.0e-4:
        verdict = "underperforms"
    else:
        verdict = "matches"
        text = (
            f"{problem_name.upper()}: {right_label} matches {left_label} "
            f"(mean HV delta {hv_delta:+.6f}; no measurable HV separation)."
        )
        if igd_delta is not None:
            if igd_delta > 1.0e-4:
                text += " IGD+ also improves."
            elif igd_delta < -1.0e-4:
                text += " IGD+ moves in the opposite direction."
            else:
                text += " IGD+ is essentially unchanged."
        return text

    if better_fraction >= 0.75:
        stability = "consistent"
    elif better_fraction >= 0.55:
        stability = "mostly positive but unstable"
    elif better_fraction >= 0.45:
        stability = "mixed and unstable"
    elif better_fraction >= 0.25:
        stability = "mostly negative and unstable"
    else:
        stability = "consistently negative"

    text = (
        f"{problem_name.upper()}: {right_label} {verdict} against {left_label} "
        f"(mean HV delta {hv_delta:+.6f}, better on {better_fraction:.0%} of paired seeds; {stability})."
    )
    if igd_delta is not None:
        if igd_delta > 1.0e-4:
            text += " IGD+ also improves."
        elif igd_delta < -1.0e-4:
            text += " IGD+ moves in the opposite direction."
        else:
            text += " IGD+ is essentially unchanged."
    return text


def print_problem_summary(problem_name: str, summary: dict[str, dict[str, float | None]]) -> None:
    print()
    print(problem_name.upper())
    print("Algorithm                 HV_mean    HV_std     IGD+_mean  Time_mean_s")
    print("----------------------------------------------------------------------")
    for variant in VARIANTS:
        row = summary[variant.label]
        print(
            f"{variant.label:<26}"
            f"{_format_metric(row['hv_mean']):<11}"
            f"{_format_metric(row['hv_std']):<11}"
            f"{_format_metric(row['igd_plus_mean']):<11}"
            f"{_format_metric(row['runtime_mean_seconds'])}"
        )


def save_plots(
    plot_dir: Path,
    problem_name: str,
    ref_front: np.ndarray,
    runs_by_algorithm: dict[str, list[dict[str, Any]]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt

        from vamos.ux.visualization import plot_hv_convergence, plot_pareto_front_2d
    except ImportError:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for label, rows in runs_by_algorithm.items():
        representative = _select_representative_run(rows)
        if representative is None:
            continue
        ax = plot_pareto_front_2d(
            np.asarray(representative["front"]["F"], dtype=float),
            labels=("f1", "f2"),
            title=f"{problem_name.upper()} {label} representative front",
            show=False,
        )
        ax.plot(ref_front[:, 0], ref_front[:, 1], color="black", linewidth=1.0, alpha=0.35)
        ax.figure.tight_layout()
        safe_label = label.lower().replace("-", "_").replace("+", "_plus_")
        front_path = plot_dir / f"{problem_name}_{safe_label}_front.png"
        ax.figure.savefig(front_path, dpi=150)
        plt.close(ax.figure)
        saved.append(_display_path(front_path))

    fig, ax = plt.subplots()
    for label, rows in runs_by_algorithm.items():
        eval_axis, hv_mean = _aggregate_mean_hv_curve(rows)
        if eval_axis.size == 0:
            continue
        plot_hv_convergence(
            eval_axis,
            hv_mean,
            ax=ax,
            label=label,
            title=f"{problem_name.upper()} mean HV convergence",
            show=False,
        )
    fig.tight_layout()
    curve_path = plot_dir / f"{problem_name}_hv_convergence.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    saved.append(_display_path(curve_path))
    return saved


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seeds))
    max_evaluations = _max_evaluations(args.population_size, args.generations)

    print(
        "Running PAVE ablation benchmark on ZDT1/ZDT2/ZDT3 "
        f"with pop_size={args.population_size}, generations={args.generations}, "
        f"seeds={args.seeds}, engine={args.engine}."
    )
    print_variant_stacks()

    if not has_moocore():
        print("IGD+ unavailable: optional dependency 'moocore' is not installed. HV and runtime will still be reported.")

    all_results: dict[str, Any] = {}
    interpretation_lines: list[str] = []

    for problem_spec in PROBLEMS:
        ref_point = get_zdt_reference_point(problem_spec.name)
        ref_front = get_zdt_reference_front(problem_spec.name, n_points=1000)
        igd_indicator = IGDPlusIndicator(reference_front=ref_front) if has_moocore() else None

        print(f"  {problem_spec.name.upper()} ...")
        problem_runs: list[dict[str, Any]] = []
        summary: dict[str, dict[str, float | None]] = {}
        runs_by_algorithm: dict[str, list[dict[str, Any]]] = {}

        for variant in VARIANTS:
            print(f"    {variant.label}")
            run_rows, run_summary = run_variant(
                problem_spec,
                variant,
                seeds,
                population_size=args.population_size,
                generations=args.generations,
                engine=args.engine,
                ref_point=ref_point,
                igd_indicator=igd_indicator,
            )
            problem_runs.extend(run_rows)
            summary[variant.label] = run_summary
            runs_by_algorithm[variant.label] = run_rows

        comparisons = {
            "new_order_vs_legacy": _hv_deltas(summary, runs_by_algorithm, "NSGAII-SBX", "NSGAII-SBX+NewOrder"),
            "pave_vs_legacy": _hv_deltas(summary, runs_by_algorithm, "NSGAII-SBX", "NSGAII-SBX+PAVE"),
            "pave_vs_new_order": _hv_deltas(summary, runs_by_algorithm, "NSGAII-SBX+NewOrder", "NSGAII-SBX+PAVE"),
            "pave_vs_directional": _hv_deltas(
                summary,
                runs_by_algorithm,
                "NSGAII-SBX+Directional",
                "NSGAII-SBX+PAVE",
            ),
        }
        interpretation_lines.append(_comparison_text(problem_spec.name, comparisons["new_order_vs_legacy"]))
        interpretation_lines.append(_comparison_text(problem_spec.name, comparisons["pave_vs_legacy"]))
        interpretation_lines.append(_comparison_text(problem_spec.name, comparisons["pave_vs_new_order"]))
        interpretation_lines.append(_comparison_text(problem_spec.name, comparisons["pave_vs_directional"]))

        plot_paths: list[str] = []
        if not args.no_plots:
            plot_paths = save_plots(args.plot_dir, problem_spec.name, ref_front, runs_by_algorithm)

        all_results[problem_spec.name] = {
            "n_var": problem_spec.n_var,
            "reference_point": ref_point,
            "igd_plus_available": igd_indicator is not None,
            "summary": summary,
            "comparisons": comparisons,
            "plots": plot_paths,
            "runs": problem_runs,
        }
        print_problem_summary(problem_spec.name, summary)

    payload = {
        "population_size": args.population_size,
        "generations": args.generations,
        "max_evaluations": max_evaluations,
        "seeds": seeds,
        "engine": args.engine,
        "variants": [
            {
                "label": variant.label,
                "crossover": variant.crossover,
                "mutation": ("polynomial", {"prob": "1/n_var", "eta": 20.0}),
                "intensification": variant.intensification,
                "stack": variant.stack,
                "repair_order": variant.repair_order,
            }
            for variant in VARIANTS
        ],
        "problems": all_results,
        "interpretation": interpretation_lines,
    }
    args.output.write_text(json.dumps(_to_builtin(payload), indent=2), encoding="utf-8")

    print()
    print("Interpretation")
    print("--------------")
    for line in interpretation_lines:
        print(f"- {line}")

    if args.no_plots:
        print()
        print("Plots skipped.")
    else:
        print()
        print(f"Plots saved under {_display_path(args.plot_dir)}")

    print(f"Saved benchmark results to {_display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Experiment-layer ablation runner example.

Build an ablation plan, convert tasks to StudyTask, and execute with run_study.

Usage:
    python examples/tuning/ablation_runner.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vamos.engine.tuning import AblationVariant, build_ablation_plan
from vamos.experiment.study import StudyTask
from vamos.experiment.study.api import run_study


def main() -> None:
    variants = [
        AblationVariant(name="baseline", label="Baseline"),
        AblationVariant(name="aos", label="AOS"),
        AblationVariant(name="tuned", label="Tuned", config_overrides={"population_size": 80}),
        AblationVariant(name="tuned_aos", label="Tuned + AOS", config_overrides={"population_size": 80}),
    ]

    plan = build_ablation_plan(
        problems=["zdt1"],
        variants=variants,
        seeds=[1, 2, 3],
        default_max_evals=2000,
        engine="numpy",
    )

    base_config = {"population_size": 50}
    algorithm = "nsgaii"
    tuned_variation = {
        "crossover": ("sbx", {"prob": 1.0, "eta": 30.0}),
        "mutation": ("pm", {"prob": "1/n", "eta": 10.0}),
    }
    variant_variations = {
        "baseline": None,
        "aos": {"adaptive_operator_selection": {"enabled": True}},
        "tuned": tuned_variation,
        "tuned_aos": {
            **tuned_variation,
            "adaptive_operator_selection": {"enabled": True},
        },
    }

    study_tasks: list[StudyTask] = []
    variant_names: list[str] = []
    for task in plan.tasks:
        overrides = task.variant.apply(base_config)
        overrides["max_evaluations"] = task.max_evals
        study_tasks.append(
            StudyTask(
                algorithm=algorithm,
                engine=task.engine or "numpy",
                problem=task.problem,
                seed=task.seed,
                config_overrides=overrides,
                nsgaii_variation=variant_variations.get(task.variant.name),
            )
        )
        variant_names.append(task.variant.name)

    results = run_study(study_tasks, mirror_output_roots=("results",))

    hv_by_variant: dict[str, list[float]] = defaultdict(list)
    for name, result in zip(variant_names, results):
        hv = result.metrics.get("hv")
        if hv is not None:
            hv_by_variant[name].append(float(hv))

    if not hv_by_variant:
        print("No hypervolume metrics available to summarize.")
        return

    medians = {name: float(np.median(vals)) for name, vals in hv_by_variant.items()}
    baseline = medians.get("baseline")
    if baseline is None:
        print("Baseline missing; medians:", medians)
        return

    deltas = {name: val - baseline for name, val in medians.items()}
    print("Median HV by variant:", medians)
    print("Delta vs baseline:", deltas)


if __name__ == "__main__":
    main()

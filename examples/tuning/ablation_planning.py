"""
Ablation planning example (same algorithm, different configurations).

Usage:
    python examples/tuning/ablation_planning.py
"""

from __future__ import annotations

from vamos.engine.tuning import AblationVariant, build_ablation_plan


def main() -> None:
    variants = [
        AblationVariant(name="baseline", label="Baseline"),
        AblationVariant(name="aos", label="AOS"),
        AblationVariant(name="tuned", label="Tuned", config_overrides={"population_size": 80}),
        AblationVariant(name="tuned_aos", label="Tuned + AOS", config_overrides={"population_size": 80}),
    ]

    plan = build_ablation_plan(
        problems=["zdt1", "dtlz2"],
        variants=variants,
        seeds=[1, 2, 3],
        default_max_evals=20000,
        engine="numpy",
    )

    print(f"Total tasks: {plan.n_tasks}")
    print("First 3 tasks:")
    for task in plan.tasks[:3]:
        print(task.as_dict())

    metrics = {
        "baseline": 0.55,
        "aos": 0.62,
        "tuned": 0.67,
        "tuned_aos": 0.71,
    }
    baseline = metrics["baseline"]
    deltas = {name: score - baseline for name, score in metrics.items()}
    print("Example deltas vs baseline:")
    print(deltas)


if __name__ == "__main__":
    main()

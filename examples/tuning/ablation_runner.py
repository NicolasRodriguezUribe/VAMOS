"""Run a small ablation as canonical, durable studies.

Each variant is one immutable StudySpec because changing an algorithm
configuration changes scientific task identity. The combined table is a
derived in-memory view of StudySummary rows and retains study/task/run IDs.

Usage:
    python examples/tuning/ablation_runner.py
"""

from __future__ import annotations

from pathlib import Path

from vamos import StudySpec, create_study, plan_study


def run_ablation(
    output_root: Path,
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    max_evaluations: int = 2_000,
    populations: tuple[tuple[str, int], ...] = (("baseline", 50), ("tuned", 80)),
) -> list[dict[str, object]]:
    """Execute each variant and return a traceable derived task table."""
    rows: list[dict[str, object]] = []
    for variant, population_size in populations:
        spec = StudySpec(
            problems=["zdt1"],
            algorithms=["nsgaii"],
            seeds=seeds,
            max_evaluations=max_evaluations,
            pop_size=population_size,
            algorithm_configs={"nsgaii": {"pop_size": population_size}},
            labels={"workflow": "ablation", "variant": variant},
        )
        planned = plan_study(spec)
        completed = create_study(spec, output=output_root / variant).run()
        if completed.plan_id != planned.plan_id:
            raise RuntimeError("Canonical planning and creation produced different plan identities.")
        for row in completed.summarize().rows:
            derived = row.as_dict()
            derived["variant"] = variant
            rows.append(derived)
    return rows


def main() -> None:
    rows = run_ablation(Path("results/ablation_demo"))
    for row in rows:
        print(
            row["variant"],
            row["seed"],
            row["state"],
            row["task_id"],
            row["selected_run_id"],
        )


if __name__ == "__main__":
    main()

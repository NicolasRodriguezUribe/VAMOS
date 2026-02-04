from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Mapping

from .types import AblationPlan, AblationTask, AblationVariant


def build_ablation_plan(
    problems: Iterable[str],
    variants: Iterable[AblationVariant],
    seeds: Iterable[int],
    default_max_evals: int,
    *,
    engine: str | None = None,
    budget_by_problem: Mapping[str, int] | None = None,
    budget_by_variant: Mapping[str, int] | None = None,
    budget_overrides: Mapping[tuple[str, str], int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AblationPlan:
    problems_seq = tuple(problems)
    variants_seq = tuple(variants)
    seeds_seq = tuple(seeds)
    if not problems_seq:
        raise ValueError("Ablation plan requires at least one problem.")
    if not variants_seq:
        raise ValueError("Ablation plan requires at least one variant.")
    if not seeds_seq:
        raise ValueError("Ablation plan requires at least one seed.")
    if default_max_evals <= 0:
        raise ValueError("default_max_evals must be a positive integer.")

    names = [variant.name for variant in variants_seq]
    if len(set(names)) != len(names):
        raise ValueError("Ablation variants must have unique names.")

    tasks: list[AblationTask] = []
    for problem in problems_seq:
        for variant in variants_seq:
            for seed in seeds_seq:
                max_evals = default_max_evals
                if budget_by_problem and problem in budget_by_problem:
                    max_evals = budget_by_problem[problem]
                if budget_by_variant and variant.name in budget_by_variant:
                    max_evals = budget_by_variant[variant.name]
                if budget_overrides and (problem, variant.name) in budget_overrides:
                    max_evals = budget_overrides[(problem, variant.name)]
                if max_evals <= 0:
                    raise ValueError(f"Invalid max_evals={max_evals} for problem={problem} variant={variant.name}.")
                tasks.append(
                    AblationTask(
                        problem=problem,
                        variant=variant,
                        seed=seed,
                        max_evals=max_evals,
                        engine=engine,
                    )
                )

    return AblationPlan(
        tasks=tuple(tasks),
        problems=problems_seq,
        variants=variants_seq,
        seeds=seeds_seq,
        default_max_evals=default_max_evals,
        engine=engine,
        metadata=dict(metadata or {}),
    )

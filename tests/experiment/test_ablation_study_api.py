from vamos.engine.tuning.api import AblationVariant, build_ablation_plan
from vamos.experiment.study.api import build_study_tasks_from_ablation_plan


def test_build_study_tasks_from_ablation_plan_sets_overrides():
    variants = [
        AblationVariant(name="baseline"),
        AblationVariant(name="tuned", config_overrides={"population_size": 80}),
    ]
    plan = build_ablation_plan(
        problems=["zdt1"],
        variants=variants,
        seeds=[1, 2],
        default_max_evals=1000,
        engine="numpy",
    )
    base_config = {"population_size": 50}
    nsgaii_variations = {"tuned": {"adaptive_operator_selection": {"enabled": True}}}

    tasks, variant_names = build_study_tasks_from_ablation_plan(
        plan,
        algorithm="nsgaii",
        base_config=base_config,
        nsgaii_variations=nsgaii_variations,
    )

    assert len(tasks) == len(plan.tasks)
    assert variant_names == [task.variant.name for task in plan.tasks]

    for task, name in zip(tasks, variant_names):
        overrides = task.config_overrides or {}
        assert overrides["max_evaluations"] == 1000
        assert task.engine == "numpy"
        if name == "tuned":
            assert overrides["population_size"] == 80
            assert task.nsgaii_variation == nsgaii_variations["tuned"]
        else:
            assert overrides["population_size"] == 50
            assert task.nsgaii_variation is None

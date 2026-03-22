from vamos.engine.tuning.api import AblationVariant, build_ablation_plan
from vamos.experiment._execution_support import VariationConfigs
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
    variations_by_variant = {
        "tuned": VariationConfigs(
            nsgaii={"crossover": ("sbx", {"prob": 0.9, "eta": 20.0})},
            moead={"aggregation": {"method": "pbi", "theta": 5.0}},
            smsemoa={"mutation": {"method": "polynomial", "prob": "1/n"}},
        )
    }

    tasks, variant_names = build_study_tasks_from_ablation_plan(
        plan,
        algorithm="nsgaii",
        base_config=base_config,
        variations_by_variant=variations_by_variant,
    )

    assert len(tasks) == len(plan.tasks)
    assert variant_names == [task.variant.name for task in plan.tasks]

    for task, name in zip(tasks, variant_names):
        overrides = task.config_overrides or {}
        assert overrides["max_evaluations"] == 1000
        assert task.engine == "numpy"
        if name == "tuned":
            assert overrides["population_size"] == 80
            assert task.variations is not None
            assert task.variations.nsgaii == variations_by_variant["tuned"].nsgaii
            assert task.variations.moead == variations_by_variant["tuned"].moead
            assert task.variations.smsemoa == variations_by_variant["tuned"].smsemoa
        else:
            assert overrides["population_size"] == 50
            assert task.variations is None

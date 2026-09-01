import json

from vamos.engine.tuning.api import AblationVariant, build_ablation_plan
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.ablation import run_ablation_plan


def test_run_ablation_plan_uses_canonical_studies_and_summaries(tmp_path):
    variants = [
        AblationVariant(name="baseline"),
        AblationVariant(name="tuned", config_overrides={"population_size": 8}),
    ]
    plan = build_ablation_plan(
        problems=["zdt1"],
        variants=variants,
        seeds=[1],
        default_max_evals=12,
        engine="numpy",
    )
    variations = {
        "tuned": VariationConfigs(nsgaii={"crossover": ("sbx", {"prob": 0.9, "eta": 20.0})}),
    }

    result = run_ablation_plan(
        plan,
        algorithm="nsgaii",
        output=tmp_path / "ablation",
        base_config={"population_size": 6, "offspring_population_size": 6},
        variations_by_variant=variations,
    )

    assert len(result.studies) == 2
    assert len(set(result.study_ids)) == 2
    assert result.study_roots == tuple(item.study.root for item in result.studies)
    assert [item.variant for item in result.studies] == ["baseline", "tuned"]
    for execution in result.studies:
        assert execution.study.status == "completed"
        assert execution.report.study_id == execution.study.study_id
        assert execution.summary.study_id == execution.study.study_id
        assert len(execution.summary.rows) == 1
        row = execution.summary.rows[0]
        assert row.selected_run_id is not None
        assert row.run_manifest_path is not None
        assert row.run_manifest_sha256 is not None
        assert row.metrics is not None
        assert row.problem_id == "vamos.problem:zdt1@1"

    derived = result.summary_rows()
    assert {row["variant"] for row in derived} == {"baseline", "tuned"}
    assert all(row["study_id"] in result.study_ids for row in derived)
    assert all(isinstance(row["hv"], float) for row in derived)
    assert all(row["hv_reference"] for row in derived)

    for execution in result.studies:
        documents = [path for path in execution.study.root.rglob("*.json") if "runs" not in path.parts]
        for document in documents:
            assert not _contains_array_key(json.loads(document.read_text(encoding="utf-8")))


def _contains_array_key(value):
    if isinstance(value, dict):
        return bool({"F", "X"} & value.keys()) or any(_contains_array_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_array_key(item) for item in value)
    return False

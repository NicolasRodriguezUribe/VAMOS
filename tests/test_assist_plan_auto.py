from __future__ import annotations

import json
from pathlib import Path

from vamos.assist.plan import create_plan
from vamos.assist.providers.mock_provider import MockPlanProvider
from vamos.engine.config.spec import validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig


def _allowed_overrides() -> set[str]:
    default_config = ExperimentConfig()
    parser = build_parser(
        default_config=default_config,
        pre_parser=build_pre_parser(),
        spec_defaults=load_spec_defaults(None),
    )
    return parser_spec_keys(parser)


def test_create_plan_auto_with_mock_provider_writes_valid_artifacts(tmp_path: Path) -> None:
    plan_dir = create_plan(
        prompt="Find a robust default setup.",
        template=None,
        problem_type="real",
        out_dir=tmp_path / "assist_plan_auto",
        mode="auto",
        provider=MockPlanProvider(),
        provider_name="mock",
    )

    prompt_path = plan_dir / "prompt.txt"
    catalog_path = plan_dir / "catalog.json"
    config_path = plan_dir / "config.json"
    metadata_path = plan_dir / "plan.json"
    assert prompt_path.is_file()
    assert catalog_path.is_file()
    assert config_path.is_file()
    assert metadata_path.is_file()

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    validate_experiment_spec(config_data, allowed_overrides=_allowed_overrides())

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "auto"
    assert metadata["provider"] == {"name": "mock"}

    auto = metadata.get("auto")
    assert isinstance(auto, dict)
    for key in ("template", "problem_type", "overrides", "warnings"):
        assert key in auto

    defaults = config_data.get("defaults")
    assert isinstance(defaults, dict)
    overrides = auto.get("overrides")
    assert isinstance(overrides, dict)
    if "max_evaluations" in defaults:
        assert defaults["max_evaluations"] == 123
        assert overrides.get("max_evaluations") == 123
    else:
        assert overrides == {}

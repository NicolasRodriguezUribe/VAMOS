from __future__ import annotations

import json

import pytest

from vamos.assist.plan import create_plan, resolve_plan_template
from vamos.experiment.cli.quickstart import available_templates
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


def test_create_plan_writes_valid_artifacts(tmp_path) -> None:
    plan_dir = tmp_path / "assist_plan"
    returned = create_plan(
        prompt="Optimize ZDT1 quickly with defaults.",
        template="demo",
        problem_type="real",
        out_dir=plan_dir,
    )
    assert returned == plan_dir

    prompt_path = plan_dir / "prompt.txt"
    catalog_path = plan_dir / "catalog.json"
    config_path = plan_dir / "config.json"
    plan_path = plan_dir / "plan.json"

    assert prompt_path.exists()
    assert catalog_path.exists()
    assert config_path.exists()
    assert plan_path.exists()

    assert prompt_path.read_text(encoding="utf-8") == "Optimize ZDT1 quickly with defaults."

    catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(plan_path.read_text(encoding="utf-8"))

    assert isinstance(catalog_data, dict)
    assert isinstance(config_data, dict)
    assert isinstance(metadata, dict)

    validate_experiment_spec(config_data, allowed_overrides=_allowed_overrides())

    for key in ("timestamp", "template", "problem_type", "schema_version", "warnings"):
        assert key in metadata
    assert metadata["schema_version"] == "1"
    assert isinstance(metadata["warnings"], list)


def test_resolve_plan_template_noninteractive_without_template_raises_clean_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_plan_template(None, is_tty=False)
    message = str(excinfo.value)
    assert "Missing --template in non-interactive mode." in message
    assert "Available templates:" in message
    assert "--template demo" in message
    for key in available_templates():
        assert key in message

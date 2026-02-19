import pytest

from vamos.engine.config.spec import allowed_override_keys, validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig


def _build_parser() -> object:
    default_config = ExperimentConfig()
    pre_parser = build_pre_parser()
    spec_defaults = load_spec_defaults(None)
    return build_parser(
        default_config=default_config,
        pre_parser=pre_parser,
        spec_defaults=spec_defaults,
    )


def _allowed_override_keys() -> set[str]:
    parser = _build_parser()
    return allowed_override_keys(parser_spec_keys(parser))


def test_validate_experiment_spec_accepts_minimal_spec() -> None:
    spec = {"version": "1"}
    validated = validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())
    assert validated["version"] == "1"


def test_validate_experiment_spec_rejects_unknown_top_level_key() -> None:
    spec = {"version": "1", "surprise": {}}
    with pytest.raises(ValueError, match="Unknown top-level keys"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_unknown_problem_key() -> None:
    spec = {"version": "1", "problems": {"zdt999": {}}}
    with pytest.raises(ValueError, match="Unknown problem key"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_invalid_operator_spec() -> None:
    spec = {"version": "1", "defaults": {"nsgaii": {"crossover": ["sbx"]}}}
    with pytest.raises(ValueError, match="tuple/list form must have exactly 2 elements"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_invalid_hv_ref_point() -> None:
    spec = {"version": "1", "stopping": {"hv_convergence": {"ref_point": ["bad"]}}}
    with pytest.raises(ValueError, match="ref_point"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_non_mapping_defaults() -> None:
    spec = {"version": "1", "defaults": 123}
    with pytest.raises(TypeError, match="defaults"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())


def test_validate_experiment_spec_rejects_invalid_aos_block() -> None:
    spec = {"version": "1", "defaults": {"nsgaii": {"adaptive_operator_selection": []}}}
    with pytest.raises(ValueError, match="adaptive_operator_selection"):
        validate_experiment_spec(spec, allowed_overrides=_allowed_override_keys())

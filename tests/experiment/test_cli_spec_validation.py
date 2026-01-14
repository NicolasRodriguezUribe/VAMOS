import pytest

from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_validation import validate_experiment_spec
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


def test_validate_experiment_spec_accepts_minimal_spec() -> None:
    parser = _build_parser()
    spec = {"version": "1"}
    validated = validate_experiment_spec(spec, parser=parser)
    assert validated["version"] == "1"


def test_validate_experiment_spec_rejects_unknown_top_level_key() -> None:
    parser = _build_parser()
    spec = {"version": "1", "surprise": {}}
    with pytest.raises(ValueError, match="Unknown top-level keys"):
        validate_experiment_spec(spec, parser=parser)


def test_validate_experiment_spec_rejects_unknown_problem_key() -> None:
    parser = _build_parser()
    spec = {"version": "1", "problems": {"zdt999": {}}}
    with pytest.raises(ValueError, match="Unknown problem key"):
        validate_experiment_spec(spec, parser=parser)


def test_validate_experiment_spec_rejects_invalid_operator_spec() -> None:
    parser = _build_parser()
    spec = {"version": "1", "defaults": {"nsgaii": {"crossover": ["sbx"]}}}
    with pytest.raises(ValueError, match="tuple/list form must have exactly 2 elements"):
        validate_experiment_spec(spec, parser=parser)


def test_validate_experiment_spec_rejects_invalid_hv_ref_point() -> None:
    parser = _build_parser()
    spec = {"version": "1", "stopping": {"hv_convergence": {"ref_point": ["bad"]}}}
    with pytest.raises(ValueError, match="ref_point"):
        validate_experiment_spec(spec, parser=parser)


def test_validate_experiment_spec_rejects_non_mapping_defaults() -> None:
    parser = _build_parser()
    spec = {"version": "1", "defaults": 123}
    with pytest.raises(TypeError, match="defaults"):
        validate_experiment_spec(spec, parser=parser)


def test_validate_experiment_spec_rejects_invalid_aos_block() -> None:
    parser = _build_parser()
    spec = {"version": "1", "defaults": {"nsgaii": {"adaptive_operator_selection": []}}}
    with pytest.raises(ValueError, match="adaptive_operator_selection"):
        validate_experiment_spec(spec, parser=parser)

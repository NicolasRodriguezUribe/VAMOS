import pytest

from vamos.engine.algorithm.factory import build_algorithm
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


@pytest.mark.smoke
def test_engine_can_run_nsgaii_minimal():
    selection = make_problem_selection("zdt1", n_var=4)
    config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=20, seed=3)
    algorithm, _ = build_algorithm(
        "nsgaii",
        "numpy",
        selection.instantiate(),
        config,
        selection_pressure=2,
    )
    termination = ("max_evaluations", 20)
    result = algorithm.run(selection.instantiate(), termination, seed=config.seed, eval_strategy=None, live_viz=None)
    assert result["F"].shape[0] > 0


def test_build_algorithm_forwards_online_control_override() -> None:
    selection = make_problem_selection("zdt1", n_var=4)
    config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=12, seed=3)
    algorithm, cfg = build_algorithm(
        "nsgaii",
        "numpy",
        selection.instantiate(),
        config,
        selection_pressure=2,
        nsgaii_variation={"online_control": {"enabled": True}},
    )
    assert algorithm.cfg["online_control"]["enabled"] is True
    assert cfg.to_dict()["online_control"]["policy"] == "hierarchical_joint"
    assert cfg.to_dict()["online_control"]["credit_model"] == "simple_improvement"

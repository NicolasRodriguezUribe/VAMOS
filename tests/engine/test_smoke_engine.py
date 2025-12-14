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
    termination = ("n_eval", 20)
    result = algorithm.run(selection.instantiate(), termination, seed=config.seed, eval_backend=None, live_viz=None)
    assert result["F"].shape[0] > 0

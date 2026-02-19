import numpy as np

from vamos.engine.tuning.racing.param_space import ParamSpace, Real
from vamos.engine.tuning.racing.random_search_tuner import RandomSearchTuner
from vamos.engine.tuning.racing.tuning_task import EvalContext, Instance, TuningTask


def test_tuning_task_eval_config_aggregates_scores():
    space = ParamSpace(params={"x": Real("x", 0.0, 1.0)})
    task = TuningTask(
        name="demo",
        param_space=space,
        instances=[Instance("p", 2)],
        seeds=[1, 2],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )

    def eval_fn(config, ctx: EvalContext) -> float:
        return ctx.seed  # deterministic per seed

    cfg = {"x": 0.5}
    score = task.eval_config(cfg, eval_fn)
    assert score == 1.5  # mean of seeds 1 and 2


def test_random_search_tuner_tracks_best():
    space = ParamSpace(params={"x": Real("x", 0.0, 1.0)})
    task = TuningTask(
        name="demo",
        param_space=space,
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )

    def eval_fn(config, ctx: EvalContext) -> float:
        return config["x"]

    tuner = RandomSearchTuner(task=task, max_trials=5, seed=0)
    best, history = tuner.run(eval_fn, verbose=False)
    assert len(history) <= 5
    assert "x" in best
    # best x should be near the maximum sampled
    scores = [t.score for t in history]
    assert np.isclose(max(scores), best["x"])

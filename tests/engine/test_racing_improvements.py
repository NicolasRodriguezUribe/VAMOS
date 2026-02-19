from types import SimpleNamespace

import numpy as np

from vamos.engine.tuning.racing.core import RacingTuner
from vamos.engine.tuning.racing.elimination import eliminate_configs
from vamos.engine.tuning.racing.param_space import Categorical, Condition, ParamSpace, Real
from vamos.engine.tuning.racing.sampler import ModelBasedSampler
from vamos.engine.tuning.racing.scenario import Scenario
from vamos.engine.tuning.racing.state import ConfigState
from vamos.engine.tuning.racing.tuning_task import Instance, TuningTask


def test_elimination_keeps_history_when_lengths_differ():
    task = SimpleNamespace(aggregator=np.mean, maximize=False)
    scenario = SimpleNamespace(
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
        min_blocks_before_elimination=1,
        alpha=0.05,
    )
    configs = [
        ConfigState(config_id=0, config={}, alive=True, scores=[0.1, 0.1, 1.0]),
        ConfigState(config_id=1, config={}, alive=True, scores=[0.5]),
    ]

    eliminated = eliminate_configs(configs, task=task, scenario=scenario)

    assert eliminated is True
    assert configs[0].alive is True
    assert configs[1].alive is False


def test_rank_elimination_waits_for_min_blocks():
    task = SimpleNamespace(aggregator=np.mean, maximize=False)
    scenario = SimpleNamespace(
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
        min_blocks_before_elimination=3,
        alpha=0.05,
    )
    configs = [
        ConfigState(config_id=0, config={}, alive=True, scores=[0.1, 0.1]),
        ConfigState(config_id=1, config={}, alive=True, scores=[0.5, 0.5]),
    ]

    eliminated = eliminate_configs(configs, task=task, scenario=scenario)

    assert eliminated is False
    assert all(c.alive for c in configs)


def test_stage_index_advances_without_elimination():
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=100,
        min_survivors=1,
        elimination_fraction=0.01,
        start_instances=1,
        use_statistical_tests=False,
        instance_order_random=False,
        seed_order_random=False,
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    def eval_fn(config, ctx):
        return 0.0

    tuner.run(eval_fn, verbose=False)

    assert tuner._stage_index == len(tuner._schedule)


def test_model_based_sampler_respects_inactive_params():
    space = ParamSpace(
        params={
            "method": Categorical("method", ["a", "b"]),
            "sigma": Real("sigma", 0.0, 1.0),
        },
        conditions=[Condition("sigma", "cfg['method'] == 'a'")],
    )
    sampler = ModelBasedSampler(space, exploration_prob=0.0, min_samples_to_model=1)
    sampler.update([{"method": "a", "sigma": 0.5}, {"method": "b"}])

    rng = np.random.default_rng(1)
    cfg = sampler.sample(rng)

    assert cfg["method"] == "b"
    assert "sigma" not in cfg

import numpy as np

from vamos.engine.tuning.core.io import filter_active_config, history_to_dict
from vamos.engine.tuning.core.param_space import ParamSpace, Categorical, Real, Condition
from vamos.engine.tuning.racing.random_search_tuner import TrialResult


def test_filter_active_config_drops_inactive_operator_params():
    param_space = ParamSpace(
        params={
            "crossover.type": Categorical(["sbx", "pcx"]),
            "crossover.sbx_eta": Real(1.0, 5.0),
            "crossover.pcx_sigma_eta": Real(0.1, 1.0),
        },
        conditions=[
            Condition("crossover.sbx_eta", "cfg['crossover.type'] == 'sbx'"),
            Condition("crossover.pcx_sigma_eta", "cfg['crossover.type'] == 'pcx'"),
        ],
    )
    cfg = {
        "crossover.type": "sbx",
        "crossover.sbx_eta": 2.0,
        "crossover.pcx_sigma_eta": 0.5,
        "population_size": 50,  # always keep globals
    }
    filtered = filter_active_config(cfg, param_space)
    assert "crossover.sbx_eta" in filtered
    assert "crossover.pcx_sigma_eta" not in filtered
    assert filtered["population_size"] == 50


def test_history_to_dict_filters_and_optionally_keeps_raw():
    param_space = ParamSpace(
        params={
            "mutation.type": Categorical(["gaussian", "cauchy"]),
            "mutation.gaussian_sigma": Real(0.01, 1.0),
            "mutation.cauchy_gamma": Real(0.01, 1.0),
        },
        conditions=[
            Condition("mutation.gaussian_sigma", "cfg['mutation.type'] == 'gaussian'"),
            Condition("mutation.cauchy_gamma", "cfg['mutation.type'] == 'cauchy'"),
        ],
    )
    cfg = {
        "mutation.type": "gaussian",
        "mutation.gaussian_sigma": 0.2,
        "mutation.cauchy_gamma": 0.3,
    }
    trial = TrialResult(trial_id=0, config=cfg, score=1.23, details={"note": "test"})
    data = history_to_dict([trial], param_space, include_raw=True)
    assert data[0]["config"]["mutation.gaussian_sigma"] == 0.2
    assert "mutation.cauchy_gamma" not in data[0]["config"]
    assert data[0]["raw_config"] == cfg

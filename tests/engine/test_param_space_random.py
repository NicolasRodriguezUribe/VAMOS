import numpy as np

from vamos.engine.tuning.racing.param_space import Categorical, Condition, Int, ParamSpace, Real


def test_param_space_sample_and_validate_with_conditions():
    space = ParamSpace(
        params={
            "a": Real("a", 0.0, 1.0),
            "b": Int("b", 1, 5),
            "c": Categorical("c", ["x", "y"]),
            "d": Real("d", 0.0, 10.0),
        },
        conditions=[Condition("d", "cfg['c'] == 'y'")],
    )
    rng = np.random.default_rng(0)
    cfg = space.sample(rng)
    space.validate(cfg)  # should not raise
    cfg_inactive = dict(cfg)
    cfg_inactive["c"] = "x"
    cfg_inactive.pop("d", None)
    space.validate(cfg_inactive)  # d inactive, so still valid

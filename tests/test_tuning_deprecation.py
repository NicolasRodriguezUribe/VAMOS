import warnings

import numpy as np

from vamos.tuning import AlgorithmConfigSpace
from vamos.tuning.core.parameter_space import Integer, Double, Categorical, ParameterDefinition
from vamos.tuning import ParamSpace


def test_algorithm_config_space_exposed():
    space = AlgorithmConfigSpace.from_template("nsgaii", "default")
    assert space.dim() > 0
    vec = np.zeros(space.dim())
    cfg = space.decode_vector(vec)
    assert hasattr(cfg, "pop_size")


def test_param_space_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _ = ParamSpace(params={})
    assert any(isinstance(w.message, DeprecationWarning) for w in caught)

import numpy as np

from vamos.engine.algorithm.config import MOEADConfig, MOEADConfigData, NSGAIIConfig, NSGAIIConfigData
from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace, Categorical, Double, Integer, ParameterDefinition


def _nsgaii_space():
    params = {
        "pop_size": ParameterDefinition(Integer(10, 20)),
        "crossover": ParameterDefinition(Categorical(["sbx"]), fixed_sub_parameters={"prob": 0.9, "eta": 20.0}),
        "mutation": ParameterDefinition(Categorical(["pm"]), fixed_sub_parameters={"prob": 0.1, "eta": 20.0}),
        "selection": ParameterDefinition(Categorical(["tournament"]), fixed_sub_parameters={"pressure": 2}),
    }
    fixed = {"survival": "nsga2", "engine": "numpy"}
    return AlgorithmConfigSpace(NSGAIIConfig, params, fixed_values=fixed)


def _moead_space():
    params = {
        "pop_size": ParameterDefinition(Integer(10, 20)),
        "neighbor_size": ParameterDefinition(Integer(2, 5)),
        "delta": ParameterDefinition(Double(0.5, 0.9)),
        "replace_limit": ParameterDefinition(Integer(1, 3)),
        "crossover": ParameterDefinition(Categorical(["sbx"]), fixed_sub_parameters={"prob": 0.9, "eta": 20.0}),
        "mutation": ParameterDefinition(Categorical(["pm"]), fixed_sub_parameters={"prob": 0.1, "eta": 20.0}),
        "aggregation": ParameterDefinition(Categorical(["tchebycheff"])),
    }
    fixed = {"engine": "numpy", "weight_vectors": {"path": None, "divisions": None}}
    return AlgorithmConfigSpace(MOEADConfig, params, fixed_values=fixed)


def test_multi_algorithm_space_decodes_specific_algorithm_configs():
    # Arrange
    nsgaii_space = _nsgaii_space()
    moead_space = _moead_space()
    space = AlgorithmConfigSpace.multi_algorithm({"nsgaii": nsgaii_space, "moead": moead_space})
    vec_nsgaii = np.concatenate(([0.0], np.full(nsgaii_space.dim(), 0.5), np.full(moead_space.dim(), 0.5)))
    vec_moead = np.concatenate(([0.9], np.full(nsgaii_space.dim(), 0.5), np.full(moead_space.dim(), 0.5)))

    # Act
    cfg1 = space.decode_vector(vec_nsgaii)
    cfg2 = space.decode_vector(vec_moead)

    # Assert
    assert isinstance(cfg1, NSGAIIConfigData)
    assert isinstance(cfg2, MOEADConfigData)

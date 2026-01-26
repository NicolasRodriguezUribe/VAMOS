import numpy as np

from vamos.engine.tuning.racing.param_space import (
    Boolean,
    Categorical,
    Int,
    Real,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.bridge import (
    build_nsgaii_config_space,
    config_from_assignment,
)


def test_param_round_trip_sampling():
    rng = np.random.default_rng(0)
    cat = Categorical("color", ["red", "blue", "green"])
    intval = Int("k", 1, 9)
    floatval = Real("p", 0.1, 0.9)
    boolp = Boolean("flag")
    cati = Categorical("size", [16, 32, 64])

    assignment = {
        "color": cat.sample(rng),
        "k": intval.sample(rng),
        "p": floatval.sample(rng),
        "flag": boolp.sample(rng),
        "size": cati.sample(rng),
    }
    vec = [
        cat.to_unit(assignment["color"]),
        intval.to_unit(assignment["k"]),
        floatval.to_unit(assignment["p"]),
        boolp.to_unit(assignment["flag"]),
        cati.to_unit(assignment["size"]),
    ]
    assert cat.from_unit(vec[0]) in cat.choices
    assert isinstance(intval.from_unit(vec[1]), int)
    assert 0.1 <= floatval.from_unit(vec[2]) <= 0.9
    assert isinstance(boolp.from_unit(vec[3]), bool)
    assert cati.from_unit(vec[4]) in cati.choices


def test_algorithm_config_space_round_trip():
    rng = np.random.default_rng(1)
    params = [Categorical("a", ["x", "y"]), Int("b", 1, 3)]
    space = AlgorithmConfigSpace("toy", params, [])
    assignment = space.sample(rng)
    vec = space.to_unit_vector(assignment)
    decoded = space.from_unit_vector(vec)
    assert decoded["a"] in ["x", "y"]
    assert 1 <= decoded["b"] <= 3


def test_nsgaii_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(2)
    space = build_nsgaii_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex")
    assert cfg.mutation[0] in ("pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform")


def test_nsgaii_archive_unbounded_disables_archive_params():
    space = build_nsgaii_config_space()
    param_space = space.to_param_space()
    cfg = {"use_external_archive": True, "archive_unbounded": True}
    assert not param_space.is_active("archive_type", cfg)
    assert not param_space.is_active("archive_size_factor", cfg)
    cfg_bounded = {"use_external_archive": True, "archive_unbounded": False}
    assert param_space.is_active("archive_type", cfg_bounded)
    assert param_space.is_active("archive_size_factor", cfg_bounded)

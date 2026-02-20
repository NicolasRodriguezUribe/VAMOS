import numpy as np

from vamos.engine.tuning.racing.bridge import (
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_permutation_config_space,
    build_ibea_binary_config_space,
    build_ibea_integer_config_space,
    build_ibea_permutation_config_space,
    build_moead_binary_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaiii_permutation_config_space,
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_permutation_config_space,
    build_smpso_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_permutation_config_space,
    build_spea2_binary_config_space,
    build_spea2_integer_config_space,
    build_spea2_permutation_config_space,
    config_from_assignment,
)
from vamos.engine.tuning.racing.config_space import AlgorithmConfigSpace
from vamos.engine.tuning.racing.param_space import (
    Boolean,
    Categorical,
    Int,
    Real,
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


def test_nsgaii_permutation_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(3)
    space = build_nsgaii_permutation_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii_permutation", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("ox", "pmx", "edge", "cycle", "position", "aex")
    assert cfg.mutation[0] in ("swap", "insert", "scramble", "inversion", "displacement", "two_opt")


def test_nsgaii_mixed_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(4)
    space = build_nsgaii_mixed_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("nsgaii_mixed", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("mixed", "uniform")
    assert cfg.mutation[0] in ("mixed", "gaussian")


def test_moead_permutation_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(5)
    space = build_moead_permutation_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("moead_permutation", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("ox", "pmx", "edge", "cycle", "position", "aex")
    assert cfg.mutation[0] in ("swap", "insert", "scramble", "inversion", "displacement", "two_opt")


def test_agemoea_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(6)
    space = build_agemoea_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("agemoea", assignment)
    assert cfg.pop_size > 0
    assert cfg.crossover[0] in ("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex")
    assert cfg.mutation[0] in (
        "pm",
        "linked_polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
    )


def test_rvea_config_space_builds_and_constructs_config():
    rng = np.random.default_rng(7)
    space = build_rvea_config_space()
    assignment = space.sample(rng)
    assignment["n_obj"] = 3
    cfg = config_from_assignment("rvea", assignment)
    assert cfg.pop_size > 0
    assert cfg.n_partitions == int(assignment["n_partitions"])
    assert cfg.crossover[0] in ("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex")
    assert cfg.mutation[0] in (
        "pm",
        "linked_polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
    )


def test_agemoea_external_archive_config():
    rng = np.random.default_rng(8)
    space = build_agemoea_config_space()
    assignment = space.sample(rng)
    assignment.update(
        {
            "use_external_archive": True,
            "archive_type": "size_cap",
            "archive_prune_policy": "crowding",
            "archive_size_factor": 2,
            "archive_epsilon": 0.01,
        }
    )
    cfg = config_from_assignment("agemoea", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity >= cfg.pop_size
    assert cfg.result_mode == "non_dominated"


def test_rvea_external_archive_config():
    rng = np.random.default_rng(9)
    space = build_rvea_config_space()
    assignment = space.sample(rng)
    assignment.update(
        {
            "n_obj": 3,
            "use_external_archive": True,
            "archive_type": "size_cap",
            "archive_prune_policy": "crowding",
            "archive_size_factor": 2,
            "archive_epsilon": 0.01,
        }
    )
    cfg = config_from_assignment("rvea", assignment)
    assert cfg.external_archive is not None
    assert cfg.external_archive.capacity >= cfg.pop_size
    assert cfg.result_mode == "non_dominated"


def test_binary_integer_config_spaces_build_and_construct_config():
    cases = [
        (
            build_nsgaii_binary_config_space,
            "nsgaii_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip"},
        ),
        (
            build_nsgaii_integer_config_space,
            "nsgaii_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm"},
        ),
        (
            build_moead_binary_config_space,
            "moead_binary",
            {"one_point", "two_point", "uniform"},
            {"bitflip"},
        ),
        (
            build_moead_integer_config_space,
            "moead_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm"},
        ),
        (
            build_nsgaiii_binary_config_space,
            "nsgaiii_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip"},
        ),
        (
            build_nsgaiii_integer_config_space,
            "nsgaiii_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm"},
        ),
        (
            build_smsemoa_binary_config_space,
            "smsemoa_binary",
            {"one_point", "two_point", "uniform"},
            {"bitflip"},
        ),
        (
            build_smsemoa_integer_config_space,
            "smsemoa_integer",
            {"uniform", "arithmetic"},
            {"reset", "creep"},
        ),
        (
            build_ibea_binary_config_space,
            "ibea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip"},
        ),
        (
            build_ibea_integer_config_space,
            "ibea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm"},
        ),
    ]

    for idx, (builder, algo, cross_set, mut_set) in enumerate(cases):
        rng = np.random.default_rng(100 + idx)
        space = builder()
        assignment = space.sample(rng)
        cfg = config_from_assignment(algo, assignment)
        assert cfg.crossover[0] in cross_set
        assert cfg.mutation[0] in mut_set


def test_new_permutation_binary_integer_builders_construct_configs():
    cases = [
        (
            build_agemoea_permutation_config_space,
            "agemoea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_agemoea_binary_config_space,
            "agemoea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            False,
        ),
        (
            build_agemoea_integer_config_space,
            "agemoea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            False,
        ),
        (
            build_rvea_permutation_config_space,
            "rvea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            True,
        ),
        (
            build_rvea_binary_config_space,
            "rvea_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            True,
        ),
        (
            build_rvea_integer_config_space,
            "rvea_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            True,
        ),
        (
            build_spea2_permutation_config_space,
            "spea2_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_spea2_binary_config_space,
            "spea2_binary",
            {"hux", "uniform", "one_point", "two_point"},
            {"bitflip", "segment_inversion"},
            False,
        ),
        (
            build_spea2_integer_config_space,
            "spea2_integer",
            {"uniform", "arithmetic", "sbx"},
            {"reset", "creep", "pm", "gaussian", "boundary"},
            False,
        ),
        (
            build_ibea_permutation_config_space,
            "ibea_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_smsemoa_permutation_config_space,
            "smsemoa_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
        (
            build_nsgaiii_permutation_config_space,
            "nsgaiii_permutation",
            {"ox", "pmx", "edge", "cycle", "position", "aex"},
            {"swap", "insert", "scramble", "inversion", "displacement", "two_opt"},
            False,
        ),
    ]
    for idx, (builder, algo, cross_set, mut_set, needs_n_obj) in enumerate(cases):
        rng = np.random.default_rng(200 + idx)
        space = builder()
        assignment = space.sample(rng)
        if needs_n_obj:
            assignment["n_obj"] = 3
        cfg = config_from_assignment(algo, assignment)
        assert cfg.crossover[0] in cross_set
        assert cfg.mutation[0] in mut_set


def test_smpso_mixed_builder_and_assignment():
    rng = np.random.default_rng(300)
    space = build_smpso_mixed_config_space()
    assignment = space.sample(rng)
    cfg = config_from_assignment("smpso_mixed", assignment)
    assert cfg.mutation[0] in {"mixed", "gaussian"}


def test_new_integer_parameter_mapping_in_assignment():
    ag_space = build_agemoea_integer_config_space()
    ag_assignment = ag_space.sample(np.random.default_rng(400))
    ag_assignment["mutation"] = "gaussian"
    ag_assignment["gaussian_sigma"] = 2.5
    ag_cfg = config_from_assignment("agemoea_integer", ag_assignment)
    assert ag_cfg.mutation[0] == "gaussian"
    assert float(ag_cfg.mutation[1]["sigma"]) == 2.5

    sp_space = build_spea2_integer_config_space()
    sp_assignment = sp_space.sample(np.random.default_rng(401))
    sp_assignment["mutation"] = "creep"
    sp_assignment["creep_step"] = 3
    sp_cfg = config_from_assignment("spea2_integer", sp_assignment)
    assert sp_cfg.mutation[0] == "creep"
    assert int(sp_cfg.mutation[1]["step"]) == 3

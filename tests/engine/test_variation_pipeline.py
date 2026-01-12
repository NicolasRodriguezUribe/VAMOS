import numpy as np

from vamos.engine.algorithm.components.variation import VariationPipeline
from vamos.operators.impl.real import VariationWorkspace


def _bounds(n_var: int) -> tuple[np.ndarray, np.ndarray]:
    xl = np.zeros(n_var, dtype=float)
    xu = np.ones(n_var, dtype=float)
    return xl, xu


def test_variation_pipeline_real_arithmetic_crossover_constructs():
    n_var = 4
    xl, xu = _bounds(n_var)
    pipeline = VariationPipeline(
        encoding="real",
        cross_method="arithmetic",
        cross_params={"prob": 1.0},
        mut_method="pm",
        mut_params={"prob": 0.5, "eta": 15.0},
        xl=xl,
        xu=xu,
        workspace=VariationWorkspace(),
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = rng.random((6, n_var))
    offspring = pipeline.produce_offspring(parents, rng)
    assert offspring.shape == parents.shape


def test_variation_pipeline_real_uniform_mutation_constructs():
    n_var = 4
    xl, xu = _bounds(n_var)
    pipeline = VariationPipeline(
        encoding="real",
        cross_method="sbx",
        cross_params={"prob": 1.0, "eta": 10.0},
        mut_method="uniform",
        mut_params={"prob": 0.5, "perturb": 0.2},
        xl=xl,
        xu=xu,
        workspace=VariationWorkspace(),
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = rng.random((6, n_var))
    offspring = pipeline.produce_offspring(parents, rng)
    assert offspring.shape == parents.shape


def test_variation_pipeline_real_undx_group_sizes():
    n_var = 4
    xl, xu = _bounds(n_var)
    pipeline = VariationPipeline(
        encoding="real",
        cross_method="undx",
        cross_params={"prob": 1.0},
        mut_method="pm",
        mut_params={"prob": 0.5, "eta": 15.0},
        xl=xl,
        xu=xu,
        workspace=VariationWorkspace(),
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = rng.random((6, n_var))  # 2 groups of 3 parents
    offspring = pipeline.produce_offspring(parents, rng)
    assert offspring.shape == (4, n_var)  # UNDX produces 2 children per group


def test_variation_pipeline_integer_sbx_crossover_constructs():
    n_var = 4
    xl = np.zeros(n_var, dtype=np.int32)
    xu = np.full(n_var, 10, dtype=np.int32)
    pipeline = VariationPipeline(
        encoding="integer",
        cross_method="sbx",
        cross_params={"prob": 1.0, "eta": 20.0},
        mut_method="reset",
        mut_params={"prob": 0.0},
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = rng.integers(0, 11, size=(5, n_var), dtype=np.int32)
    offspring = pipeline.produce_offspring(parents, rng)

    assert offspring.shape == parents.shape


def test_variation_pipeline_binary_odd_parent_count():
    n_var = 8
    xl = np.zeros(n_var, dtype=np.int8)
    xu = np.ones(n_var, dtype=np.int8)
    pipeline = VariationPipeline(
        encoding="binary",
        cross_method="one_point",
        cross_params={"prob": 1.0},
        mut_method="bitflip",
        mut_params={"prob": 0.5},
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = rng.integers(0, 2, size=(5, n_var), dtype=np.int8)
    offspring = pipeline.produce_offspring(parents, rng)

    assert offspring.shape == parents.shape


def test_variation_pipeline_permutation_odd_parent_count():
    n_var = 6
    xl = np.zeros(n_var, dtype=np.int32)
    xu = np.full(n_var, n_var - 1, dtype=np.int32)
    pipeline = VariationPipeline(
        encoding="permutation",
        cross_method="ox",
        cross_params={"prob": 1.0},
        mut_method="swap",
        mut_params={"prob": 0.5},
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=None,
        problem=None,
    )

    rng = np.random.default_rng(0)
    parents = np.vstack([rng.permutation(n_var) for _ in range(5)]).astype(np.int32, copy=False)
    offspring = pipeline.produce_offspring(parents, rng)

    assert offspring.shape == parents.shape


def test_variation_pipeline_mixed_odd_parent_count():
    class DummyMixedProblem:
        mixed_spec = {
            "real_idx": np.array([0, 1], dtype=int),
            "int_idx": np.array([], dtype=int),
            "cat_idx": np.array([], dtype=int),
            "real_lower": np.zeros(2, dtype=float),
            "real_upper": np.ones(2, dtype=float),
        }

    n_var = 4
    xl = np.zeros(n_var, dtype=float)
    xu = np.ones(n_var, dtype=float)
    pipeline = VariationPipeline(
        encoding="mixed",
        cross_method="mixed",
        cross_params={"prob": 1.0},
        mut_method="mixed",
        mut_params={"prob": 0.5},
        xl=xl,
        xu=xu,
        workspace=None,
        repair_cfg=None,
        problem=DummyMixedProblem(),
    )

    rng = np.random.default_rng(0)
    parents = rng.random((5, n_var))
    offspring = pipeline.produce_offspring(parents, rng)

    assert offspring.shape == parents.shape

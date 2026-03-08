import numpy as np
from numpy.testing import assert_allclose

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.operators.impl.real import PAVEIntensification, VariationWorkspace
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _bounds(n_var: int) -> tuple[np.ndarray, np.ndarray]:
    xl = np.zeros(n_var, dtype=float)
    xu = np.ones(n_var, dtype=float)
    return xl, xu


def test_pave_workspace_fallback_without_workspace_context() -> None:
    n_var = 5
    xl, xu = _bounds(n_var)
    rng = np.random.default_rng(0)
    parents = rng.random((8, n_var))
    offspring = rng.random((8, n_var))
    operator = PAVEIntensification(
        k_neighbors=3,
        alpha=0.35,
        beta=0.2,
        lambda_distance=0.5,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
        workspace=None,
    )

    refined = operator(offspring, np.random.default_rng(1), parents=parents)

    assert refined.shape == offspring.shape
    assert operator.last_population_context_available is False
    assert operator.last_objective_context_available is False
    assert operator.last_fallback_used is True


def test_pave_small_population_handles_large_k_neighbors() -> None:
    n_var = 4
    xl, xu = _bounds(n_var)
    offspring = np.array(
        [
            [0.2, 0.3, 0.4, 0.5],
            [0.7, 0.8, 0.9, 0.1],
        ],
        dtype=float,
    )
    parents = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.6, 0.7, 0.8, 0.9],
        ],
        dtype=float,
    )
    operator = PAVEIntensification(
        k_neighbors=10,
        alpha=0.25,
        beta=0.1,
        lambda_distance=0.5,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
    )

    refined = operator(offspring, np.random.default_rng(2), parents=parents)

    assert refined.shape == offspring.shape
    assert np.isfinite(refined).all()


def test_pave_is_deterministic_for_fixed_seed() -> None:
    n_var = 6
    xl, xu = _bounds(n_var)
    rng = np.random.default_rng(3)
    parents = rng.random((10, n_var))
    offspring = rng.random((10, n_var))
    operator = PAVEIntensification(
        k_neighbors=4,
        alpha=0.3,
        beta=0.15,
        lambda_distance=0.4,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
    )

    offspring_a = operator(offspring, np.random.default_rng(11), parents=parents)
    offspring_b = operator(offspring, np.random.default_rng(11), parents=parents)

    assert_allclose(offspring_a, offspring_b)


def test_pave_probability_mask_can_skip_intensification() -> None:
    n_var = 5
    xl, xu = _bounds(n_var)
    rng = np.random.default_rng(4)
    parents = rng.random((6, n_var))
    offspring = rng.random((6, n_var))
    operator = PAVEIntensification(
        k_neighbors=3,
        alpha=0.35,
        beta=0.2,
        lambda_distance=0.5,
        prob_intensification=0.0,
        lower=xl,
        upper=xu,
    )

    refined = operator(offspring, np.random.default_rng(12), parents=parents)

    assert_allclose(refined, offspring)


def test_pave_shape_and_finiteness_safety() -> None:
    n_var = 7
    xl, xu = _bounds(n_var)
    rng = np.random.default_rng(5)
    parents = rng.uniform(-10.0, 10.0, size=(12, n_var))
    offspring = rng.uniform(-10.0, 10.0, size=(12, n_var))
    operator = PAVEIntensification(
        k_neighbors=5,
        alpha=0.35,
        beta=0.2,
        lambda_distance=0.6,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
    )

    refined = operator(offspring, np.random.default_rng(13), parents=parents)

    assert refined.shape == (12, n_var)
    assert np.isfinite(refined).all()


def test_pave_uses_bound_workspace_population_when_available() -> None:
    n_var = 3
    xl, xu = _bounds(n_var)
    workspace = VariationWorkspace()
    workspace.bind_population(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.2, 0.3, 0.4],
                [0.9, 0.1, 0.8],
            ],
            dtype=float,
        ),
        np.array(
            [
                [1.0, 4.0],
                [4.0, 1.0],
                [1.5, 2.0],
                [2.5, 1.5],
            ],
            dtype=float,
        ),
    )
    parents = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.3, 0.2, 0.4],
            [0.8, 0.2, 0.7],
        ],
        dtype=float,
    )
    offspring = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.25, 0.2, 0.35],
            [0.75, 0.25, 0.65],
        ],
        dtype=float,
    )

    with_workspace = PAVEIntensification(
        k_neighbors=2,
        alpha=0.3,
        beta=0.1,
        lambda_distance=0.7,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
        workspace=workspace,
    )
    without_workspace = PAVEIntensification(
        k_neighbors=2,
        alpha=0.3,
        beta=0.1,
        lambda_distance=0.7,
        prob_intensification=1.0,
        lower=xl,
        upper=xu,
        workspace=None,
    )

    refined_workspace = with_workspace(offspring, np.random.default_rng(21), parents=parents)
    refined_fallback = without_workspace(offspring, np.random.default_rng(21), parents=parents)

    assert refined_workspace.shape == offspring.shape
    assert not np.allclose(refined_workspace, refined_fallback)
    assert with_workspace.last_population_context_available is True
    assert with_workspace.last_objective_context_available is True
    assert with_workspace.last_fallback_used is False
    assert without_workspace.last_fallback_used is True


def test_nsgaii_executes_pave_as_intensification_with_live_workspace_context() -> None:
    pop_size = 12
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .intensification(
            "pave",
            prob=1.0,
            k_neighbors=4,
            alpha=0.35,
            beta=0.2,
            lambda_distance=0.5,
        )
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)
    algo._initialize_run(
        problem,
        termination=("max_evaluations", pop_size * 2),
        seed=7,
        eval_strategy=None,
        live_viz=None,
    )

    st = algo._st
    assert st is not None
    assert st.variation.cross_method == "sbx"
    assert st.variation.intensification_method == "pave"
    assert st.variation.intensification_op is not None

    pave_op = getattr(st.variation.intensification_op, "_operator", None)
    assert isinstance(pave_op, PAVEIntensification)
    assert pave_op.workspace is st.variation_workspace
    assert pave_op.call_count == 0

    X_off = algo.ask()

    assert X_off.shape[0] == pop_size
    assert st.variation_workspace.population is not None
    assert st.variation_workspace.objectives is not None
    assert pave_op.call_count == 1
    assert pave_op.last_population_context_available is True
    assert pave_op.last_objective_context_available is True
    assert pave_op.last_fallback_used is False

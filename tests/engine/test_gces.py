from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vamos.algorithms import NSGAIIConfig, available_algorithms, resolve_algorithm
from vamos.api import optimize
from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.algorithm.gces import (
    GCES,
    GCESNoComp,
    GCESNoGeo,
    NSGA2CurvGap,
    NSGA2Farthest,
    NSGA2GapFill,
    NSGA2HVFarthest,
    NSGA2HVRefFarthest,
    NSGA2RefCoverFarthest,
    NSGA2SectorFarthest,
)
from vamos.engine.algorithm.gces import selector as gces_selector
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.zdt1 import ZDT1Problem as ZDT1

ALL_GCES_ALGORITHMS = ("gces", "gces_nocomp", "gces_nogeo")
ALL_NSGA2_HOSTED_2D_ALGORITHMS = ALL_GCES_ALGORITHMS + (
    "nsga2_farthest",
    "nsga2_gapfill",
    "nsga2_curvgap",
    "nsga2_hvfarthest",
    "nsga2_refcover_farthest",
    "nsga2_hvref_farthest",
)
ALL_NSGA2_HOSTED_ALGORITHMS = ALL_NSGA2_HOSTED_2D_ALGORITHMS + ("nsga2_sector_farthest",)
NSGA2_HOSTED_TYPES = {
    "gces": GCES,
    "gces_nocomp": GCESNoComp,
    "gces_nogeo": GCESNoGeo,
    "nsga2_farthest": NSGA2Farthest,
    "nsga2_gapfill": NSGA2GapFill,
    "nsga2_curvgap": NSGA2CurvGap,
    "nsga2_hvfarthest": NSGA2HVFarthest,
    "nsga2_refcover_farthest": NSGA2RefCoverFarthest,
    "nsga2_hvref_farthest": NSGA2HVRefFarthest,
    "nsga2_sector_farthest": NSGA2SectorFarthest,
}


class _ConstrainedProblem:
    n_var = 2
    n_obj = 2
    n_constraints = 1
    xl = 0.0
    xu = 1.0
    encoding = "real"

    def evaluate(self, X, out) -> None:  # pragma: no cover - should never be reached
        raise AssertionError("GCES phase-1 validation should reject constrained problems before evaluation.")


def _gces_cfg(
    *,
    pop_size: int = 20,
    offspring_size: int = 20,
    steady_state: bool = False,
    track_genealogy: bool = False,
) -> NSGAIIConfig:
    builder = NSGAIIConfig.builder()
    builder.pop_size(pop_size)
    builder.offspring_size(offspring_size)
    builder.selection("tournament", size=2)
    builder.crossover("sbx", prob=1.0, eta=20.0)
    builder.mutation("pm", prob=0.1, eta=20.0)
    if steady_state:
        builder.steady_state(True)
    if track_genealogy:
        builder.track_genealogy(True)
    return builder.build()


def _normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(F, dtype=float)
    spans = nadir - ideal
    valid = spans > 0.0
    if np.any(valid):
        normalized[:, valid] = (F[:, valid] - ideal[valid]) / spans[valid]
    return normalized


@pytest.mark.parametrize("algorithm", ALL_NSGA2_HOSTED_ALGORITHMS)
def test_available_algorithms_include_nsga2_hosted_variants(algorithm: str) -> None:
    assert algorithm in available_algorithms()


@pytest.mark.parametrize("algorithm", ALL_NSGA2_HOSTED_ALGORITHMS)
def test_resolve_algorithm_nsga2_hosted_variants_return_expected_builder(algorithm: str) -> None:
    cfg = NSGAIIConfig.default(pop_size=6, n_var=4)
    builder = resolve_algorithm(algorithm)
    algo = builder(cfg.to_dict(), resolve_kernel("numpy"))

    assert isinstance(algo, NSGA2_HOSTED_TYPES[algorithm])


@pytest.mark.parametrize("algorithm", ALL_NSGA2_HOSTED_2D_ALGORITHMS)
@pytest.mark.smoke
def test_optimize_nsga2_hosted_variants_run_on_unconstrained_problem(algorithm: str) -> None:
    problem = ZDT1(n_var=10)
    result = optimize(problem, algorithm=algorithm, max_evaluations=200, pop_size=20, seed=42)

    assert result.F is not None
    assert result.F.shape[0] > 0
    assert result.meta["algorithm"] == algorithm


@pytest.mark.smoke
def test_optimize_gces_runs_with_nsgaii_config() -> None:
    cfg = NSGAIIConfig.default(pop_size=20, n_var=10)
    problem = ZDT1(n_var=10)

    result = optimize(problem, algorithm="gces", algorithm_config=cfg, max_evaluations=200, seed=42)

    assert result.F is not None
    assert result.F.shape[0] > 0
    assert result.meta["algorithm"] == "gces"


@pytest.mark.parametrize("algorithm_name", ALL_NSGA2_HOSTED_2D_ALGORITHMS)
@pytest.mark.smoke
def test_build_algorithm_nsga2_hosted_variants_returns_instance_and_runs(algorithm_name: str) -> None:
    selection = make_problem_selection("zdt1", n_var=4)
    config = ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=20, seed=3)
    algorithm, cfg = build_algorithm(
        algorithm_name,
        "numpy",
        selection.instantiate(),
        config,
        selection_pressure=2,
    )

    assert isinstance(algorithm, NSGA2_HOSTED_TYPES[algorithm_name])
    assert isinstance(cfg, NSGAIIConfig)

    result = algorithm.run(selection.instantiate(), ("max_evaluations", 20), seed=config.seed, eval_strategy=None, live_viz=None)
    assert result["F"].shape[0] > 0


@pytest.mark.parametrize("algorithm", ("nsgaii",) + ALL_NSGA2_HOSTED_2D_ALGORITHMS)
@pytest.mark.smoke
def test_zcat1_smoke_runs_for_full_ablation_family(algorithm: str) -> None:
    cfg = NSGAIIConfig.default(pop_size=12, n_var=30)

    result = optimize(
        "zcat1",
        algorithm=algorithm,
        algorithm_config=cfg,
        max_evaluations=48,
        seed=1,
        engine="numpy",
        n_var=30,
        n_obj=2,
    )

    assert result.F is not None
    assert result.F.shape[1] == 2
    assert np.isfinite(result.F).all()
    assert result.meta["algorithm"] == algorithm


@pytest.mark.smoke
def test_nsga2_sector_farthest_runs_on_three_objective_zcat_problem() -> None:
    cfg = NSGAIIConfig.default(pop_size=12, n_var=30)

    result = optimize(
        "zcat1",
        algorithm="nsga2_sector_farthest",
        algorithm_config=cfg,
        max_evaluations=48,
        seed=1,
        engine="numpy",
        n_var=30,
        n_obj=3,
    )

    assert result.F is not None
    assert result.F.shape[1] == 3
    assert np.isfinite(result.F).all()
    assert result.meta["algorithm"] == "nsga2_sector_farthest"


@pytest.mark.smoke
def test_build_algorithm_nsga2_sector_farthest_returns_instance_and_runs() -> None:
    selection = make_problem_selection("zcat1", n_var=30, n_obj=3)
    config = ExperimentConfig(population_size=12, offspring_population_size=12, max_evaluations=36, seed=3)
    algorithm, cfg = build_algorithm(
        "nsga2_sector_farthest",
        "numpy",
        selection.instantiate(),
        config,
        selection_pressure=2,
    )

    assert isinstance(algorithm, NSGA2SectorFarthest)
    assert isinstance(cfg, NSGAIIConfig)

    result = algorithm.run(selection.instantiate(), ("max_evaluations", 36), seed=config.seed, eval_strategy=None, live_viz=None)
    assert result["F"].shape[0] > 0


def test_gces_rejects_constrained_problems() -> None:
    with pytest.raises(ValueError, match="does not support constrained problems"):
        optimize(_ConstrainedProblem(), algorithm="gces", max_evaluations=20, pop_size=6, seed=1)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (_gces_cfg(pop_size=20, offspring_size=20, steady_state=True), "does not support steady-state mode"),
        (_gces_cfg(pop_size=20, offspring_size=10), "does not support incremental replacement"),
    ],
)
def test_gces_rejects_incremental_modes(cfg: NSGAIIConfig, message: str) -> None:
    problem = ZDT1(n_var=10)
    with pytest.raises(ValueError, match=message):
        optimize(problem, algorithm="gces", algorithm_config=cfg, max_evaluations=50, seed=1)


def test_gces_rejects_moocore_engine() -> None:
    problem = ZDT1(n_var=10)
    algo = GCES(NSGAIIConfig.default(pop_size=20, n_var=10).to_dict(), kernel=SimpleNamespace(name="moocore"))

    with pytest.raises(ValueError, match="does not support the moocore engine"):
        algo.run(problem, ("max_evaluations", 20), seed=1)


def test_gces_rejects_genealogy_tracking() -> None:
    cfg = _gces_cfg(pop_size=20, offspring_size=20, track_genealogy=True)
    problem = ZDT1(n_var=10)

    with pytest.raises(ValueError, match="does not support genealogy tracking"):
        optimize(problem, algorithm="gces", algorithm_config=cfg, max_evaluations=50, seed=1)


def test_select_split_front_gces_trivial_cases() -> None:
    F_split = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ideal = np.array([0.0, 0.0])
    nadir = np.array([2.0, 2.0])
    rng = np.random.default_rng(123)

    assert np.array_equal(gces_selector.select_split_front_gces(F_split, 0, ideal, nadir, rng), np.array([], dtype=int))
    assert np.array_equal(gces_selector.select_split_front_gces(F_split, 3, ideal, nadir, rng), np.array([0, 1, 2], dtype=int))
    assert np.array_equal(gces_selector.select_split_front_gces(F_split, 4, ideal, nadir, rng), np.array([0, 1, 2], dtype=int))


def test_select_split_front_gces_handles_zero_span_coordinate() -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 1.0], dtype=float)
    nadir = np.array([3.0, 1.0], dtype=float)

    selected = gces_selector.select_split_front_gces(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 3], dtype=int))


def test_select_split_front_gces_identical_points_tie_breaks_by_index() -> None:
    F_split = np.ones((4, 2), dtype=float)
    ideal = np.array([1.0, 1.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected_one = gces_selector.select_split_front_gces(F_split, 1, ideal, nadir, np.random.default_rng(0))
    selected_three = gces_selector.select_split_front_gces(F_split, 3, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_one, np.array([0], dtype=int))
    assert np.array_equal(selected_three, np.array([0, 1, 2], dtype=int))


def test_select_split_front_gces_is_deterministic_on_synthetic_split_front() -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [5.0, 5.0],
            [5.0, 5.3],
            [5.0, 5.6],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([5.0, 5.6], dtype=float)

    selected = gces_selector.select_split_front_gces(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 2, 4], dtype=int))


def test_select_split_front_gces_is_rng_independent_under_ties() -> None:
    F_split = np.ones((5, 2), dtype=float)
    ideal = np.array([1.0, 1.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected_a = gces_selector.select_split_front_gces(F_split, 2, ideal, nadir, np.random.default_rng(1))
    selected_b = gces_selector.select_split_front_gces(F_split, 2, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([0, 1], dtype=int))


def test_select_split_front_nsga2_farthest_trivial_cases() -> None:
    F_split = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ideal = np.array([0.0, 0.0])
    nadir = np.array([2.0, 2.0])
    rng = np.random.default_rng(123)

    assert np.array_equal(
        gces_selector.select_split_front_nsga2_farthest(F_split, 0, ideal, nadir, rng),
        np.array([], dtype=int),
    )
    assert np.array_equal(
        gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, rng),
        np.array([0, 1, 2], dtype=int),
    )
    assert np.array_equal(
        gces_selector.select_split_front_nsga2_farthest(F_split, 4, ideal, nadir, rng),
        np.array([0, 1, 2], dtype=int),
    )


def test_select_split_front_nsga2_farthest_preserves_extremes_when_possible() -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.4, 0.4],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected = gces_selector.select_split_front_nsga2_farthest(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1], dtype=int))


def test_select_split_front_nsga2_farthest_resolves_too_many_extremes_deterministically() -> None:
    F_split = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0, 1.0], dtype=float)

    selected = gces_selector.select_split_front_nsga2_farthest(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1], dtype=int))


def test_select_split_front_nsga2_farthest_is_rng_independent_under_ties() -> None:
    F_split = np.ones((5, 2), dtype=float)
    ideal = np.array([1.0, 1.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected_a = gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, np.random.default_rng(1))
    selected_b = gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([0, 1, 2], dtype=int))


def test_select_split_front_nsga2_farthest_adds_farthest_point_after_extremes() -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.4, 0.4],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected = gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1, 2], dtype=int))


@pytest.mark.parametrize(
    "selector",
    [
        gces_selector.select_split_front_nsga2_hvfarthest,
        gces_selector.select_split_front_nsga2_refcover_farthest,
        gces_selector.select_split_front_nsga2_hvref_farthest,
    ],
)
def test_select_split_front_score_farthest_variants_preserve_extremes_when_possible(selector) -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.4, 0.4],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected = selector(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1], dtype=int))


def test_select_split_front_nsga2_sector_farthest_preserves_extremes_when_possible() -> None:
    F_split = np.array(
        [
            [0.0, 0.8, 0.8],
            [0.8, 0.0, 0.8],
            [0.8, 0.8, 0.0],
            [0.4, 0.4, 0.4],
            [0.3, 0.5, 0.6],
        ],
        dtype=float,
    )
    ideal = np.zeros(3, dtype=float)
    nadir = np.array([0.8, 0.8, 0.8], dtype=float)

    selected = gces_selector.select_split_front_nsga2_sector_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1, 2], dtype=int))


@pytest.mark.parametrize(
    "selector",
    [
        gces_selector.select_split_front_nsga2_hvfarthest,
        gces_selector.select_split_front_nsga2_refcover_farthest,
        gces_selector.select_split_front_nsga2_hvref_farthest,
    ],
)
def test_select_split_front_score_farthest_variants_are_rng_independent_under_ties(selector) -> None:
    F_split = np.ones((5, 2), dtype=float)
    ideal = np.array([1.0, 1.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected_a = selector(F_split, 3, ideal, nadir, np.random.default_rng(1))
    selected_b = selector(F_split, 3, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([0, 1, 2], dtype=int))


def test_select_split_front_nsga2_sector_farthest_is_rng_independent_under_ties() -> None:
    F_split = np.ones((5, 3), dtype=float)
    ideal = np.ones(3, dtype=float)
    nadir = np.ones(3, dtype=float)

    selected_a = gces_selector.select_split_front_nsga2_sector_farthest(F_split, 4, ideal, nadir, np.random.default_rng(1))
    selected_b = gces_selector.select_split_front_nsga2_sector_farthest(F_split, 4, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([0, 1, 2, 3], dtype=int))


@pytest.mark.parametrize(
    ("selector", "selector_name"),
    [
        (gces_selector.select_split_front_nsga2_hvfarthest, "nsga2_hvfarthest"),
        (gces_selector.select_split_front_nsga2_refcover_farthest, "nsga2_refcover_farthest"),
        (gces_selector.select_split_front_nsga2_hvref_farthest, "nsga2_hvref_farthest"),
    ],
)
def test_select_split_front_score_farthest_variants_require_two_or_three_objectives(selector, selector_name: str) -> None:
    F_split = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 3.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    ideal = np.zeros(4, dtype=float)
    nadir = np.array([1.0, 1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match=f"{selector_name} currently supports only 2- or 3-objective split fronts"):
        selector(F_split, 2, ideal, nadir, np.random.default_rng(0))


def test_select_split_front_nsga2_sector_farthest_requires_three_objectives() -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    ideal = np.zeros(2, dtype=float)
    nadir = np.ones(2, dtype=float)

    with pytest.raises(ValueError, match="nsga2_sector_farthest currently supports only 3-objective split fronts"):
        gces_selector.select_split_front_nsga2_sector_farthest(F_split, 2, ideal, nadir, np.random.default_rng(0))


def test_select_split_front_nsga2_hvfarthest_changes_choice_relative_to_farthest() -> None:
    F_split = np.array(
        [
            [0.1488, 0.9726],
            [0.8899, 0.8224],
            [0.48, 0.2324],
            [0.8019, 0.9235],
            [0.2661, 0.5389],
            [0.4428, 0.9310],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    selected_farthest = gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))
    selected_hvfarthest = gces_selector.select_split_front_nsga2_hvfarthest(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected_farthest, np.array([0, 1, 2], dtype=int))
    assert np.array_equal(selected_hvfarthest, np.array([0, 2, 4], dtype=int))


def test_select_split_front_nsga2_refcover_farthest_changes_choice_relative_to_farthest() -> None:
    F_split = np.array(
        [
            [0.3443, 0.4303],
            [0.9661, 0.5622],
            [0.2589, 0.2417],
            [0.8881, 0.2259],
            [0.1246, 0.2883],
            [0.5861, 0.5541],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    selected_farthest = gces_selector.select_split_front_nsga2_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))
    selected_refcover = gces_selector.select_split_front_nsga2_refcover_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected_farthest, np.array([1, 3, 4], dtype=int))
    assert np.array_equal(selected_refcover, np.array([3, 4, 5], dtype=int))


def test_select_split_front_nsga2_hvref_farthest_uses_mixed_score() -> None:
    F_split = np.array(
        [
            [0.3297, 0.7884],
            [0.3032, 0.4535],
            [0.1340, 0.4031],
            [0.2035, 0.2623],
            [0.7504, 0.2804],
            [0.4852, 0.9807],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    selected_hv = gces_selector.select_split_front_nsga2_hvfarthest(F_split, 3, ideal, nadir, np.random.default_rng(0))
    selected_ref = gces_selector.select_split_front_nsga2_refcover_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))
    selected_mix = gces_selector.select_split_front_nsga2_hvref_farthest(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected_hv, np.array([2, 3, 5], dtype=int))
    assert np.array_equal(selected_ref, np.array([0, 2, 3], dtype=int))
    assert np.array_equal(selected_mix, np.array([2, 3, 4], dtype=int))


def test_select_split_front_nsga2_sector_farthest_changes_choice_relative_to_farthest() -> None:
    F_split = np.array(
        [
            [0.7991, 0.1190, 0.3201],
            [0.8151, 0.4213, 0.7422],
            [0.4691, 0.6557, 0.5781],
            [0.0433, 0.2934, 0.1586],
            [0.5722, 0.4908, 0.4900],
            [0.1078, 0.5850, 0.1835],
            [0.9594, 0.7795, 0.6686],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    selected_farthest = gces_selector.select_split_front_nsga2_farthest(F_split, 4, ideal, nadir, np.random.default_rng(0))
    selected_sector = gces_selector.select_split_front_nsga2_sector_farthest(F_split, 4, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected_farthest, np.array([0, 3, 4, 6], dtype=int))
    assert np.array_equal(selected_sector, np.array([0, 2, 3, 6], dtype=int))


@pytest.mark.numba
@pytest.mark.parametrize(
    "selector",
    [
        gces_selector.select_split_front_nsga2_farthest,
        gces_selector.select_split_front_nsga2_hvfarthest,
        gces_selector.select_split_front_nsga2_hvref_farthest,
    ],
)
def test_score_farthest_selectors_match_numpy_fallback_when_numba_is_available(selector) -> None:
    pytest.importorskip("numba")
    F_split = np.array(
        [
            [0.1488, 0.9726, 0.4120],
            [0.8899, 0.8224, 0.1330],
            [0.4800, 0.2324, 0.7610],
            [0.8019, 0.9235, 0.2980],
            [0.2661, 0.5389, 0.6100],
            [0.4428, 0.9310, 0.5210],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    original_disabled = gces_selector._SELECTOR_NUMBA_DISABLED
    original_pair = gces_selector._PAIRWISE_DISTANCES_JIT
    original_refdist = gces_selector._REFERENCE_DISTANCE_MATRIX_JIT
    original_refgain = gces_selector._REFERENCE_COVER_GAINS_JIT
    try:
        gces_selector._SELECTOR_NUMBA_DISABLED = False
        selected_numba = selector(F_split, 4, ideal, nadir, np.random.default_rng(0))

        gces_selector._SELECTOR_NUMBA_DISABLED = True
        gces_selector._PAIRWISE_DISTANCES_JIT = None
        gces_selector._REFERENCE_DISTANCE_MATRIX_JIT = None
        gces_selector._REFERENCE_COVER_GAINS_JIT = None
        selected_numpy = selector(F_split, 4, ideal, nadir, np.random.default_rng(0))
    finally:
        gces_selector._SELECTOR_NUMBA_DISABLED = original_disabled
        gces_selector._PAIRWISE_DISTANCES_JIT = original_pair
        gces_selector._REFERENCE_DISTANCE_MATRIX_JIT = original_refdist
        gces_selector._REFERENCE_COVER_GAINS_JIT = original_refgain

    assert np.array_equal(selected_numba, selected_numpy)


@pytest.mark.numba
def test_nsga2_sector_farthest_matches_numpy_fallback_when_numba_is_available() -> None:
    pytest.importorskip("numba")
    F_split = np.array(
        [
            [0.7991, 0.1190, 0.3201],
            [0.8151, 0.4213, 0.7422],
            [0.4691, 0.6557, 0.5781],
            [0.0433, 0.2934, 0.1586],
            [0.5722, 0.4908, 0.4900],
            [0.1078, 0.5850, 0.1835],
            [0.9594, 0.7795, 0.6686],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    original_disabled = gces_selector._SELECTOR_NUMBA_DISABLED
    original_pair = gces_selector._PAIRWISE_DISTANCES_JIT
    original_refdist = gces_selector._REFERENCE_DISTANCE_MATRIX_JIT
    original_refgain = gces_selector._REFERENCE_COVER_GAINS_JIT
    try:
        gces_selector._SELECTOR_NUMBA_DISABLED = False
        selected_numba = gces_selector.select_split_front_nsga2_sector_farthest(
            F_split,
            4,
            ideal,
            nadir,
            np.random.default_rng(0),
        )

        gces_selector._SELECTOR_NUMBA_DISABLED = True
        gces_selector._PAIRWISE_DISTANCES_JIT = None
        gces_selector._REFERENCE_DISTANCE_MATRIX_JIT = None
        gces_selector._REFERENCE_COVER_GAINS_JIT = None
        selected_numpy = gces_selector.select_split_front_nsga2_sector_farthest(
            F_split,
            4,
            ideal,
            nadir,
            np.random.default_rng(0),
        )
    finally:
        gces_selector._SELECTOR_NUMBA_DISABLED = original_disabled
        gces_selector._PAIRWISE_DISTANCES_JIT = original_pair
        gces_selector._REFERENCE_DISTANCE_MATRIX_JIT = original_refdist
        gces_selector._REFERENCE_COVER_GAINS_JIT = original_refgain

    assert np.array_equal(selected_numba, selected_numpy)


@pytest.mark.parametrize(
    "selector",
    [
        gces_selector.select_split_front_nsga2_gapfill,
        gces_selector.select_split_front_nsga2_curvgap,
    ],
)
def test_select_split_front_gap_variants_preserve_extremes_when_possible(selector) -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.4, 0.4],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected = selector(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1], dtype=int))


@pytest.mark.parametrize(
    "selector",
    [
        gces_selector.select_split_front_nsga2_gapfill,
        gces_selector.select_split_front_nsga2_curvgap,
    ],
)
def test_select_split_front_gap_variants_are_rng_independent_under_ties(selector) -> None:
    F_split = np.ones((5, 2), dtype=float)
    ideal = np.array([1.0, 1.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected_a = selector(F_split, 3, ideal, nadir, np.random.default_rng(1))
    selected_b = selector(F_split, 3, ideal, nadir, np.random.default_rng(999))

    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([0, 1, 2], dtype=int))


@pytest.mark.parametrize(
    ("selector", "selector_name"),
    [
        (gces_selector.select_split_front_nsga2_gapfill, "nsga2_gapfill"),
        (gces_selector.select_split_front_nsga2_curvgap, "nsga2_curvgap"),
    ],
)
def test_select_split_front_gap_variants_require_two_objectives(selector, selector_name: str) -> None:
    F_split = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match=f"{selector_name} currently supports only 2-objective split fronts"):
        selector(F_split, 2, ideal, nadir, np.random.default_rng(0))


def test_select_split_front_nsga2_gapfill_prefers_best_gap_splitter() -> None:
    F_split = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([1.0, 1.0], dtype=float)

    selected = gces_selector.select_split_front_nsga2_gapfill(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 2, 4], dtype=int))


def test_select_split_front_nsga2_curvgap_changes_choice_relative_to_gapfill() -> None:
    F_split = np.array(
        [
            [0.03, 0.91],
            [0.15, 0.51],
            [0.47, 0.69],
            [0.62, 0.65],
            [0.63, 0.30],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    selected_gapfill = gces_selector.select_split_front_nsga2_gapfill(F_split, 3, ideal, nadir, np.random.default_rng(0))
    selected_curvgap = gces_selector.select_split_front_nsga2_curvgap(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected_gapfill, np.array([0, 2, 4], dtype=int))
    assert np.array_equal(selected_curvgap, np.array([0, 1, 4], dtype=int))


def test_select_split_front_gces_nocomp_skips_component_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    def fail_components(*args, **kwargs):
        raise AssertionError("gces_nocomp should not call component detection.")

    monkeypatch.setattr(gces_selector, "_build_components", fail_components)

    selected = gces_selector.select_split_front_gces_nocomp(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert selected.size == 2
    assert np.array_equal(selected, np.array([0, 3], dtype=int))


def test_select_split_front_gces_nogeo_skips_geodesic_shortest_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
            [3.2, 3.0],
            [3.4, 3.0],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)

    def fail_geodesic(*args, **kwargs):
        raise AssertionError("gces_nogeo should not compute geodesic shortest paths.")

    monkeypatch.setattr(gces_selector, "_all_pairs_shortest_paths", fail_geodesic)

    selected = gces_selector.select_split_front_gces_nogeo(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert selected.size == 3
    assert np.array_equal(selected, np.array([0, 3, 5], dtype=int))


def test_select_split_front_gces_zero_total_weight_allocation_fallback_is_deterministic() -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([10.0, 10.0], dtype=float)

    selected = gces_selector.select_split_front_gces(F_split, 3, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 1, 2], dtype=int))


def test_select_split_front_gces_more_components_than_slots_keeps_heaviest_components() -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=float,
    )
    ideal = np.array([0.0, 0.0], dtype=float)
    nadir = np.array([20.0, 20.0], dtype=float)

    selected = gces_selector.select_split_front_gces(F_split, 2, ideal, nadir, np.random.default_rng(0))

    assert np.array_equal(selected, np.array([0, 4], dtype=int))


def test_selector_component_detection_handles_no_cut_and_cut_cases() -> None:
    F_no_cut = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ],
        dtype=float,
    )
    ideal_no_cut = F_no_cut.min(axis=0)
    nadir_no_cut = F_no_cut.max(axis=0)
    normalized_no_cut = _normalize(F_no_cut, ideal_no_cut, nadir_no_cut)
    distances_no_cut = gces_selector._pairwise_distances(normalized_no_cut)
    mst_no_cut = gces_selector._build_complete_mst(distances_no_cut)
    components_no_cut = gces_selector._build_components(len(F_no_cut), distances_no_cut, mst_no_cut)

    F_cut = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
        ],
        dtype=float,
    )
    ideal_cut = F_cut.min(axis=0)
    nadir_cut = F_cut.max(axis=0)
    normalized_cut = _normalize(F_cut, ideal_cut, nadir_cut)
    distances_cut = gces_selector._pairwise_distances(normalized_cut)
    mst_cut = gces_selector._build_complete_mst(distances_cut)
    components_cut = gces_selector._build_components(len(F_cut), distances_cut, mst_cut)

    assert [component.indices.tolist() for component in components_no_cut] == [[0, 1, 2, 3]]
    assert [component.indices.tolist() for component in components_cut] == [[0, 1, 2], [3]]


def test_selector_component_detection_covers_size_one_and_size_two_components() -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)
    normalized = _normalize(F_split, ideal, nadir)
    distances = gces_selector._pairwise_distances(normalized)
    mst = gces_selector._build_complete_mst(distances)
    components = gces_selector._build_components(len(F_split), distances, mst)
    component_sizes = sorted(component.indices.size for component in components)

    assert component_sizes == [1, 1, 4]


def test_selector_component_detection_covers_size_two_components() -> None:
    F_split = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=float,
    )
    ideal = F_split.min(axis=0)
    nadir = F_split.max(axis=0)
    normalized = _normalize(F_split, ideal, nadir)
    distances = gces_selector._pairwise_distances(normalized)
    mst = gces_selector._build_complete_mst(distances)
    components = gces_selector._build_components(len(F_split), distances, mst)
    component_sizes = sorted(component.indices.size for component in components)

    assert component_sizes == [2, 2]


@pytest.mark.parametrize(
    ("algorithm_cls", "selector_name"),
    [
        (GCES, "select_split_front_gces"),
        (GCESNoComp, "select_split_front_gces_nocomp"),
        (GCESNoGeo, "select_split_front_gces_nogeo"),
        (NSGA2Farthest, "select_split_front_nsga2_farthest"),
        (NSGA2GapFill, "select_split_front_nsga2_gapfill"),
        (NSGA2CurvGap, "select_split_front_nsga2_curvgap"),
        (NSGA2HVFarthest, "select_split_front_nsga2_hvfarthest"),
        (NSGA2RefCoverFarthest, "select_split_front_nsga2_refcover_farthest"),
        (NSGA2HVRefFarthest, "select_split_front_nsga2_hvref_farthest"),
        (NSGA2SectorFarthest, "select_split_front_nsga2_sector_farthest"),
    ],
)
def test_gces_family_tell_uses_split_front_selector_path(
    monkeypatch: pytest.MonkeyPatch,
    algorithm_cls: type[GCES],
    selector_name: str,
) -> None:
    problem = ZDT1(n_var=4)
    algo = algorithm_cls(NSGAIIConfig.default(pop_size=6, n_var=4).to_dict(), kernel=resolve_kernel("numpy"))
    _live, eval_backend, _max_eval, _n_eval, _hv = algo._initialize_run(
        problem,
        ("max_evaluations", 12),
        seed=7,
        eval_strategy=SerialEvalBackend(),
        live_viz=None,
    )

    called = {"selector": False}

    def fake_selector(F_split, slots, ideal, nadir, rng):
        called["selector"] = True
        assert F_split.shape[0] == 6
        assert slots == 3
        assert ideal.shape == nadir.shape == (2,)
        return np.array([0, 2, 4], dtype=int)

    def fake_ranking(F):
        n = F.shape[0]
        ranks = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=int)
        assert n == ranks.size
        return ranks, np.zeros(n, dtype=float)

    def fail_survival(*args, **kwargs):
        raise AssertionError("GCES.tell() should not call backend nsga2_survival().")

    X_off = algo.ask()
    eval_off = eval_backend.evaluate(X_off, problem)

    monkeypatch.setattr(gces_selector, selector_name, fake_selector)
    monkeypatch.setattr(algo.kernel, "nsga2_ranking", fake_ranking)
    monkeypatch.setattr(algo.kernel, "nsga2_survival", fail_survival)

    algo.tell(eval_off)

    assert called["selector"] is True
    assert algo._st is not None
    assert algo._st.F.shape[0] == 6


def test_baseline_nsgaii_still_uses_backend_survival(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = ZDT1(n_var=4)
    algo = NSGAII(NSGAIIConfig.default(pop_size=6, n_var=4).to_dict(), kernel=resolve_kernel("numpy"))
    _live, eval_backend, _max_eval, _n_eval, _hv = algo._initialize_run(
        problem,
        ("max_evaluations", 12),
        seed=5,
        eval_strategy=SerialEvalBackend(),
        live_viz=None,
    )

    original_survival = algo.kernel.nsga2_survival
    called = {"survival": False}

    def wrapped_survival(*args, **kwargs):
        called["survival"] = True
        return original_survival(*args, **kwargs)

    monkeypatch.setattr(algo.kernel, "nsga2_survival", wrapped_survival)

    X_off = algo.ask()
    eval_off = eval_backend.evaluate(X_off, problem)
    algo.tell(eval_off)

    assert called["survival"] is True

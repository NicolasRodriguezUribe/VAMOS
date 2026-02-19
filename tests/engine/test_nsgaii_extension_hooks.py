from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vamos.engine.algorithm.nsgaii.ask_tell as ask_tell_module
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaii.injection import ImmigrationManager
from vamos.foundation.kernel.numpy_backend import NumPyKernel


class IdentityProblem:
    def __init__(self, n_var: int = 2) -> None:
        self.n_var = n_var
        self.n_obj = 2
        self.xl = np.zeros(n_var, dtype=float)
        self.xu = np.ones(n_var, dtype=float)
        self.encoding = "continuous"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.asarray(X[:, :2], dtype=float).copy()


def _base_cfg() -> dict[str, object]:
    return {
        "pop_size": 12,
        "offspring_size": 12,
        "crossover": ("sbx", {"prob": 0.9, "eta": 15.0}),
        "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
        "selection": ("tournament", {"pressure": 2}),
    }


def test_ask_parent_pool_respects_filter_and_non_breeding(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    algo = NSGAII(cfg, NumPyKernel())
    problem = IdentityProblem()
    algo._initialize_run(problem, termination=("max_evaluations", 24), seed=0, eval_strategy=None, live_viz=None)
    st = algo._st
    assert st is not None
    st.parent_selection_filter = lambda _state, _ranks, _crowding: np.array([0, 1, 2], dtype=int)
    st.non_breeding_indices = np.array([1], dtype=int)

    observed: dict[str, np.ndarray | None] = {"candidate_indices": None}
    original_pool = ask_tell_module.build_mating_pool

    def _spy_build_mating_pool(
        kernel: object,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        parent_count: int,
        parents_per_group: int,
        sel_method: str,
        candidate_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        observed["candidate_indices"] = (
            np.asarray(candidate_indices, dtype=int).copy()
            if candidate_indices is not None
            else None
        )
        return original_pool(
            kernel,
            ranks,
            crowding,
            pressure,
            rng,
            parent_count,
            parents_per_group,
            sel_method,
            candidate_indices=candidate_indices,
        )

    monkeypatch.setattr(ask_tell_module, "build_mating_pool", _spy_build_mating_pool)
    _ = algo.ask()

    assert observed["candidate_indices"] is not None
    np.testing.assert_array_equal(observed["candidate_indices"], np.array([0, 2], dtype=int))


@pytest.mark.parametrize(
    ("breeding_mode", "expect_zero_mating"),
    [
        ("normal", False),
        ("exclude", True),
        ("freeze_reinsert", True),
    ],
)
def test_immigration_modes_control_mating_participation(
    breeding_mode: str,
    expect_zero_mating: bool,
) -> None:
    cfg = _base_cfg()
    cfg["immigration"] = {
        "enabled": True,
        "start_generation": 0,
        "every": 1,
        "max_in_pop": 1,
        "max_insert_per_event": 1,
        "breeding_mode": breeding_mode,
        "static_candidates": [
            {
                "X": np.array([0.0, 0.0], dtype=float),
                "F": np.array([0.0, 0.0], dtype=float),
            }
        ],
    }
    algo = NSGAII(cfg, NumPyKernel())
    _ = algo.run(IdentityProblem(), termination=("max_evaluations", 120), seed=11)
    st = algo._st
    assert st is not None
    assert st.immigration_manager is not None
    totals = st.immigration_manager.totals()
    assert totals.inserted == 1
    if expect_zero_mating:
        assert totals.mating_participation == 0
    else:
        assert totals.mating_participation > 0


def test_generation_callback_population_archive_payload_is_rich() -> None:
    cfg = _base_cfg()
    captured: dict[str, object] = {}

    def _on_generation(payload: dict[str, object]) -> bool:
        captured["payload"] = payload
        population = payload["population"]
        assert isinstance(population, dict)
        pop_x = population["X"]
        assert isinstance(pop_x, np.ndarray)
        pop_x[:] = np.nan
        return True

    cfg["external_archive"] = {"capacity": 24}
    cfg["live_callback_mode"] = "population_archive"
    cfg["generation_callback"] = _on_generation
    cfg["generation_callback_copy"] = True

    algo = NSGAII(cfg, NumPyKernel())
    _ = algo.run(IdentityProblem(), termination=("max_evaluations", 120), seed=3)
    st = algo._st
    assert st is not None
    assert np.isfinite(st.X).all()

    payload = captured.get("payload")
    assert isinstance(payload, dict)
    population = payload.get("population")
    nondominated = payload.get("nondominated")
    archive = payload.get("archive")
    stats = payload.get("stats")

    assert isinstance(population, dict)
    assert isinstance(population.get("X"), np.ndarray)
    assert population["X"].shape[0] == int(cfg["pop_size"])
    assert isinstance(nondominated, dict)
    assert isinstance(nondominated.get("X"), np.ndarray)
    assert isinstance(archive, dict)
    assert isinstance(archive.get("X"), np.ndarray)
    assert isinstance(stats, dict)
    assert isinstance(stats.get("archive"), dict)


def test_freeze_reinsert_reinjects_missing_active_candidate() -> None:
    state = SimpleNamespace(
        X=np.array(
            [
                [0.8, 0.8],
                [0.6, 0.6],
                [0.4, 0.4],
            ],
            dtype=float,
        ),
        F=np.array(
            [
                [0.8, 0.8],
                [0.6, 0.6],
                [0.4, 0.4],
            ],
            dtype=float,
        ),
        G=None,
        constraint_mode="none",
        rng=np.random.default_rng(0),
        non_breeding_indices=np.zeros(0, dtype=int),
    )

    manager = ImmigrationManager(
        {
            "enabled": True,
            "start_generation": 0,
            "every": 1,
            "max_in_pop": 1,
            "max_insert_per_event": 1,
            "breeding_mode": "freeze_reinsert",
            "static_candidates": [
                {
                    "X": np.array([0.0, 0.0], dtype=float),
                    "F": np.array([0.0, 0.0], dtype=float),
                }
            ],
        }
    )
    problem = IdentityProblem()
    kernel = NumPyKernel()

    changed_0 = manager.apply_generation(generation=0, state=state, problem=problem, kernel=kernel)
    assert changed_0
    active_idx = manager.active_indices()
    assert active_idx.size == 1

    removed_idx = int(active_idx[0])
    state.X[removed_idx] = np.array([0.95, 0.95], dtype=float)
    state.F[removed_idx] = np.array([0.95, 0.95], dtype=float)

    changed_1 = manager.apply_generation(generation=1, state=state, problem=problem, kernel=kernel)
    assert changed_1
    active_idx_after = manager.active_indices()
    assert active_idx_after.size == 1
    np.testing.assert_array_equal(state.non_breeding_indices, active_idx_after)

    has_reinserted = np.any(np.all(np.isclose(state.X, np.array([0.0, 0.0]), atol=1e-9), axis=1))
    assert has_reinserted

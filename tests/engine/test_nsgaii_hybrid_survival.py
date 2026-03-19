from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vamos.engine.algorithm.nsgaii.ask_tell as ask_tell_module
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaii.survival import (
    archive_aware_nsga2_survival,
    compute_archive_novelty_scores,
    normalize_scores,
    select_hybrid_split_front,
    supports_archive_hybrid_survival,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _builder(pop_size: int):
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )


def _run_algo(cfg: NSGAIIConfig, problem: object, *, seed: int, max_eval: int) -> tuple[dict[str, object], NSGAII]:
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    result = algo.run(problem, termination=("max_evaluations", max_eval), seed=seed)
    return result, algo


class _ConstrainedBiObjective:
    n_var = 2
    n_obj = 2
    n_constr = 1
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    encoding = "real"

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0] + X[:, 1]])
        out["G"] = (X[:, 0] + X[:, 1] - 1.25)[:, None]


def test_normalize_scores_handles_infinities_and_identical_values_without_nans():
    scores = np.array([np.inf, 2.0, 2.0, 0.0], dtype=float)
    normalized = normalize_scores(scores)

    assert np.isfinite(normalized).all()
    assert normalized[0] == pytest.approx(1.0)
    assert normalized[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(normalize_scores(np.array([5.0, 5.0], dtype=float)), np.array([0.5, 0.5]))
    np.testing.assert_allclose(normalize_scores(np.array([np.inf, np.inf], dtype=float)), np.array([0.5, 0.5]))


def test_compute_archive_novelty_scores_falls_back_for_missing_or_small_archive():
    split_F = np.array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]], dtype=float)

    raw_none, norm_none = compute_archive_novelty_scores(split_F, None, k=1)
    assert raw_none is None and norm_none is None

    raw_small, norm_small = compute_archive_novelty_scores(split_F, np.array([[0.0, 4.0]], dtype=float), k=2)
    assert raw_small is None and norm_small is None


def test_select_hybrid_split_front_prefers_historically_less_crowded_candidate():
    split_F = np.array(
        [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )
    archive_F = np.array(
        [
            [1.0, 3.0],
            [1.02, 2.98],
            [1.05, 2.95],
        ],
        dtype=float,
    )

    selected_idx, scores = select_hybrid_split_front(split_F, 3, archive_F=archive_F, alpha=0.5, k=1)

    assert bool(scores["used_archive"])
    assert scores["split_front_mode"] == "archive"
    assert scores["novelty_fallback_reason"] is None
    assert 0 in selected_idx and 4 in selected_idx
    assert 3 in selected_idx
    assert 1 not in selected_idx


def test_small_archive_fallback_matches_local_only_selection():
    split_F = np.array(
        [
            [0.0, 4.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )

    selected_none, scores_none = select_hybrid_split_front(split_F, 3, archive_F=None, alpha=0.25, k=3)
    selected_small, scores_small = select_hybrid_split_front(
        split_F,
        3,
        archive_F=np.array([[0.0, 4.0], [4.0, 0.0]], dtype=float),
        alpha=0.25,
        k=3,
    )

    assert not bool(scores_none["used_archive"])
    assert not bool(scores_small["used_archive"])
    assert scores_none["split_front_mode"] == "local_only"
    assert scores_none["novelty_fallback_reason"] == "archive_missing"
    assert scores_small["split_front_mode"] == "local_only"
    assert scores_small["novelty_fallback_reason"] == "archive_too_small"
    np.testing.assert_array_equal(selected_none, selected_small)


def test_archive_aware_nsga2_survival_preserves_complete_front_acceptance_and_population_size():
    kernel = NumPyKernel()
    X = np.arange(4, dtype=float).reshape(-1, 1)
    F = np.array(
        [
            [0.0, 0.6],
            [0.6, 0.0],
            [0.4, 1.0],
            [0.6, 0.8],
        ],
        dtype=float,
    )
    X_off = np.arange(4, 7, dtype=float).reshape(-1, 1)
    F_off = np.array(
        [
            [0.8, 0.6],
            [1.0, 0.4],
            [1.2, 0.2],
        ],
        dtype=float,
    )
    archive_F = np.array([[0.6, 0.8], [0.61, 0.79], [0.62, 0.78]], dtype=float)

    _, _, selected_idx = archive_aware_nsga2_survival(
        kernel,
        X,
        F,
        X_off,
        F_off,
        5,
        archive_F=archive_F,
        alpha=0.5,
        k=1,
        return_indices=True,
    )

    assert selected_idx.shape == (5,)
    assert 0 in selected_idx and 1 in selected_idx


def test_support_check_rejects_incremental_and_constrained_modes():
    incremental_state = SimpleNamespace(archive_mode="hybrid_survival", incremental_mode=True, G=None)
    constrained_state = SimpleNamespace(archive_mode="hybrid_survival", incremental_mode=False, G=np.zeros((1, 1), dtype=float))

    assert supports_archive_hybrid_survival(incremental_state) == (False, "incremental_mode")
    assert supports_archive_hybrid_survival(constrained_state) == (False, "constraints")


def test_archive_mode_off_uses_standard_path(monkeypatch: pytest.MonkeyPatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("archive_aware_nsga2_survival should not be called when archive_mode='off'.")

    monkeypatch.setattr(ask_tell_module, "archive_aware_nsga2_survival", _unexpected_call)
    result, algo = _run_algo(_builder(10).build(), ZDT1Problem(n_var=8), seed=7, max_eval=40)
    assert result["F"].shape[0] > 0
    assert algo._st is not None
    assert algo._st.archive_hybrid_last_status == "inactive"
    diagnostics = result["archive_diagnostics"]
    assert diagnostics["archive_mode"] == "off"
    assert diagnostics["execution_mode"] == "standard"
    assert diagnostics["survival_path"] == "standard"
    assert diagnostics["archive_present"] is False


def test_passive_mode_still_uses_standard_path(monkeypatch: pytest.MonkeyPatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("archive_aware_nsga2_survival should not be called when archive_mode='passive'.")

    monkeypatch.setattr(ask_tell_module, "archive_aware_nsga2_survival", _unexpected_call)
    result, algo = _run_algo(_builder(10).archive_mode("passive").build(), ZDT1Problem(n_var=8), seed=8, max_eval=40)
    assert "archive" in result
    assert algo._st is not None
    assert algo._st.archive_hybrid_last_status == "inactive"
    diagnostics = result["archive_diagnostics"]
    assert diagnostics["archive_mode"] == "passive"
    assert diagnostics["execution_mode"] == "passive_archive"
    assert diagnostics["survival_path"] == "standard"


def test_hybrid_mode_falls_back_to_standard_survival_for_constrained_run():
    pop_size = 12
    max_eval = pop_size * 4
    problem = _ConstrainedBiObjective()

    baseline_result, _ = _run_algo(_builder(pop_size).constraint_mode("feasibility").build(), problem, seed=5, max_eval=max_eval)
    hybrid_result, hybrid_algo = _run_algo(
        _builder(pop_size).constraint_mode("feasibility").archive_mode("hybrid_survival").build(),
        problem,
        seed=5,
        max_eval=max_eval,
    )

    np.testing.assert_allclose(hybrid_result["population"]["X"], baseline_result["population"]["X"])
    np.testing.assert_allclose(hybrid_result["population"]["F"], baseline_result["population"]["F"])
    np.testing.assert_allclose(hybrid_result["F"], baseline_result["F"])
    assert "archive" in hybrid_result
    assert hybrid_algo._st is not None
    assert hybrid_algo._st.archive_hybrid_last_status == "fallback"
    assert hybrid_algo._st.archive_hybrid_fallback_reason == "constraints"
    diagnostics = hybrid_result["archive_diagnostics"]
    assert diagnostics["execution_mode"] == "hybrid_fallback"
    assert diagnostics["survival_path"] == "standard"
    assert diagnostics["hybrid_fallback_reason"] == "constraints"
    assert diagnostics["archive_present"] is True


def test_hybrid_mode_falls_back_to_standard_survival_for_incremental_run():
    pop_size = 12
    max_eval = pop_size * 4
    problem = ZDT1Problem(n_var=8)

    baseline_result, _ = _run_algo(_builder(pop_size).offspring_size(1).build(), problem, seed=13, max_eval=max_eval)
    hybrid_result, hybrid_algo = _run_algo(
        _builder(pop_size).offspring_size(1).archive_mode("hybrid_survival").build(),
        problem,
        seed=13,
        max_eval=max_eval,
    )

    np.testing.assert_allclose(hybrid_result["population"]["X"], baseline_result["population"]["X"])
    np.testing.assert_allclose(hybrid_result["population"]["F"], baseline_result["population"]["F"])
    np.testing.assert_allclose(hybrid_result["F"], baseline_result["F"])
    assert hybrid_algo._st is not None
    assert hybrid_algo._st.archive_hybrid_last_status == "fallback"
    assert hybrid_algo._st.archive_hybrid_fallback_reason == "incremental_mode"
    diagnostics = hybrid_result["archive_diagnostics"]
    assert diagnostics["execution_mode"] == "hybrid_fallback"
    assert diagnostics["survival_path"] == "standard"
    assert diagnostics["hybrid_fallback_reason"] == "incremental_mode"


def test_hybrid_mode_smoke_run_exercises_hybrid_path():
    pop_size = 12
    max_eval = pop_size * 5
    cfg = (
        _builder(pop_size)
        .archive_mode("hybrid_survival")
        .archive_hybrid_alpha(0.4)
        .archive_hybrid_k(1)
        .build()
    )

    result, algo = _run_algo(cfg, ZDT1Problem(n_var=8), seed=17, max_eval=max_eval)

    assert algo._st is not None
    assert algo._st.archive_hybrid_last_status == "hybrid"
    assert "archive" in result
    assert result["population"]["F"].shape == (pop_size, 2)
    assert np.isfinite(result["population"]["F"]).all()
    assert np.isfinite(result["archive"]["F"]).all()
    assert np.isfinite(result["archive"]["subset"]["F"]).all()
    diagnostics = result["archive_diagnostics"]
    assert diagnostics["archive_mode"] == "hybrid_survival"
    assert diagnostics["execution_mode"] == "hybrid_survival"
    assert diagnostics["survival_path"] == "hybrid"
    assert diagnostics["hybrid_status"] == "hybrid"
    assert diagnostics["hybrid_generations"] > 0
    assert diagnostics["archive_present"] is True
    assert diagnostics["archive_size"] == result["archive"]["size"]


def test_hybrid_mode_local_only_split_fallback_is_traceable():
    pop_size = 10
    max_eval = pop_size * 4
    cfg = (
        _builder(pop_size)
        .archive_mode("hybrid_survival")
        .archive_hybrid_k(10_000)
        .build()
    )

    result, algo = _run_algo(cfg, ZDT1Problem(n_var=8), seed=23, max_eval=max_eval)

    assert algo._st is not None
    diagnostics = result["archive_diagnostics"]
    assert diagnostics["execution_mode"] == "hybrid_survival"
    assert diagnostics["survival_path"] == "hybrid"
    assert diagnostics["hybrid_split_front_mode"] == "local_only"
    assert diagnostics["hybrid_split_front_reason"] == "archive_too_small"
    assert diagnostics["hybrid_local_only_generations"] > 0
    assert diagnostics["hybrid_archive_reference_generations"] == 0
    assert diagnostics["archive_present"] is True
    assert result["archive"]["size"] < 10_000


def test_hybrid_mode_rejects_bounded_external_archive():
    with pytest.raises(ValueError, match="archive_mode='hybrid_survival'"):
        _builder(10).external_archive(capacity=20, pruning="crowding").archive_mode("hybrid_survival").build()

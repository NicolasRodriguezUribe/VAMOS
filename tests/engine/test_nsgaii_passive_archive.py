from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
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


def _run(cfg: NSGAIIConfig, problem: object, *, seed: int, max_eval: int) -> dict[str, object]:
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    return algo.run(problem, termination=("max_evaluations", max_eval), seed=seed)


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


def test_passive_archive_mode_preserves_baseline_population_and_front():
    pop_size = 14
    max_eval = pop_size * 5
    problem = ZDT1Problem(n_var=8)

    baseline = _run(_builder(pop_size).build(), problem, seed=21, max_eval=max_eval)
    passive = _run(_builder(pop_size).archive_mode("passive").build(), problem, seed=21, max_eval=max_eval)

    assert "archive" not in baseline
    np.testing.assert_allclose(passive["population"]["X"], baseline["population"]["X"])
    np.testing.assert_allclose(passive["population"]["F"], baseline["population"]["F"])
    np.testing.assert_allclose(passive["X"], baseline["X"])
    np.testing.assert_allclose(passive["F"], baseline["F"])
    baseline_diag = baseline["archive_diagnostics"]
    passive_diag = passive["archive_diagnostics"]
    assert baseline_diag["archive_mode"] == "off"
    assert baseline_diag["execution_mode"] == "standard"
    assert baseline_diag["survival_path"] == "standard"
    assert passive_diag["archive_mode"] == "passive"
    assert passive_diag["execution_mode"] == "passive_archive"
    assert passive_diag["survival_path"] == "standard"
    assert passive_diag["archive_present"] is True


def test_passive_archive_mode_exports_full_archive_and_default_subset():
    pop_size = 10
    max_eval = pop_size * 6
    result = _run(_builder(pop_size).archive_mode("passive").build(), ZDT1Problem(n_var=8), seed=9, max_eval=max_eval)

    archive = result["archive"]
    assert isinstance(archive, dict)
    subset = archive["subset"]
    assert isinstance(subset, dict)

    assert archive["F"].shape[0] > 0
    assert archive["X"].shape[0] == archive["F"].shape[0]
    assert archive["size"] == archive["F"].shape[0]
    assert archive["size"] >= result["F"].shape[0]

    assert subset["selector"] == "crowding"
    assert subset["size"] == subset["F"].shape[0]
    assert subset["X"].shape[0] == subset["size"]
    assert subset["indices"].shape[0] == subset["size"]
    assert subset["size"] == min(pop_size, archive["size"])
    assert subset["size"] <= archive["size"]

    diagnostics = result["archive_diagnostics"]
    assert diagnostics["archive_mode"] == "passive"
    assert diagnostics["execution_mode"] == "passive_archive"
    assert diagnostics["archive_present"] is True
    assert diagnostics["archive_size"] == archive["size"]
    assert diagnostics["archive_subset_size"] == subset["size"]
    assert diagnostics["archive_subset_selector"] == subset["selector"]


def test_passive_archive_mode_exports_constraints_and_configured_subset_size():
    pop_size = 12
    max_eval = pop_size * 4
    cfg = (
        _builder(pop_size)
        .constraint_mode("feasibility")
        .archive_mode("passive")
        .archive_subset_size(3)
        .build()
    )

    result = _run(cfg, _ConstrainedBiObjective(), seed=5, max_eval=max_eval)
    archive = result["archive"]
    assert isinstance(archive, dict)
    assert archive["G"] is not None
    assert archive["G"].shape[0] == archive["size"]

    subset = archive["subset"]
    assert isinstance(subset, dict)
    assert subset["size"] == min(3, archive["size"])
    assert subset["G"] is not None
    assert subset["G"].shape[0] == subset["size"]
    diagnostics = result["archive_diagnostics"]
    assert diagnostics["archive_size"] == archive["size"]
    assert diagnostics["archive_subset_size"] == subset["size"]


def test_passive_archive_mode_rejects_bounded_external_archive():
    with pytest.raises(ValueError, match="archive_mode='passive'"):
        _builder(10).external_archive(capacity=20, pruning="crowding").archive_mode("passive").build()

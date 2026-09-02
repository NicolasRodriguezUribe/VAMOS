from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
import pytest

import vamos.engine.algorithm.registry as registry_mod
from vamos.engine.algorithm.config import AlgorithmConfigMapping, GenericAlgorithmConfig
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol


class _FakeEntryPoint:
    def __init__(self, name: str, builder: Callable[..., Any]) -> None:
        self.name = name
        self._builder = builder

    def load(self) -> Callable[..., Any]:
        return self._builder


class _UnusedProblem:
    n_var = 1
    n_obj = 2
    n_constraints = 0
    xl: float | int | np.ndarray[Any, Any] = 0.0
    xu: float | int | np.ndarray[Any, Any] = 1.0
    encoding: EncodingLike = "real"

    def evaluate(self, X: Any, out: Any) -> None:
        raise AssertionError("The plugin fixture must not evaluate its problem.")


def test_algorithm_registry_loads_entry_point_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    class _PluginAlgorithm:
        def __init__(self, cfg: dict[str, object]) -> None:
            self.cfg = cfg

        def run(
            self,
            problem: ProblemProtocol,
            termination: tuple[str, Any] = ("max_evaluations", 1),
            seed: int = 0,
            eval_strategy: Any | None = None,
            live_viz: Any | None = None,
        ) -> dict[str, Any]:
            answer = self.cfg["answer"]
            assert isinstance(answer, (int, float))
            return {"F": np.array([[float(answer), 0.0]]), "X": None}

    def _build_plugin_algorithm(
        cfg: AlgorithmConfigMapping,
        kernel: KernelBackend | None = None,
    ) -> registry_mod.AlgorithmLike:
        return _PluginAlgorithm(dict(cfg))

    monkeypatch.setattr(registry_mod, "_ALGORITHMS", None)
    monkeypatch.setattr(registry_mod, "_ALGORITHM_PLUGINS_LOADED", False)
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: [_FakeEntryPoint("_pytest_plugin_algo", _build_plugin_algorithm)]
        if kwargs.get("group") == "vamos.algorithms"
        else [],
    )

    loaded = registry_mod.discover_algorithm_plugins(force=True)
    registry = registry_mod.get_algorithms_registry()

    assert loaded == ("_pytest_plugin_algo",)
    algorithm = registry["_pytest_plugin_algo"](GenericAlgorithmConfig({"answer": 42}).to_dict(), None)
    result = algorithm.run(problem=_UnusedProblem())
    np.testing.assert_allclose(result["F"], np.array([[42.0, 0.0]]))

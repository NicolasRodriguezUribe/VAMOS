from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

import vamos.engine.algorithm.registry as registry_mod
from vamos.engine.algorithm.config import GenericAlgorithmConfig


class _FakeEntryPoint:
    def __init__(self, name: str, builder: Callable[..., Any]) -> None:
        self.name = name
        self._builder = builder

    def load(self) -> Callable[..., Any]:
        return self._builder


class _FakeEntryPoints:
    def __init__(self, entry_points: list[_FakeEntryPoint]) -> None:
        self._entry_points = entry_points

    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        if group == "vamos.algorithms":
            return self._entry_points
        return []


def test_algorithm_registry_loads_entry_point_plugins(monkeypatch):
    class _PluginAlgorithm:
        def __init__(self, cfg: dict[str, object]) -> None:
            self.cfg = cfg

        def run(self, problem, termination=("max_evaluations", 1), seed=0, eval_strategy=None, live_viz=None):
            return {"F": np.array([[float(self.cfg["answer"]), 0.0]]), "X": None}

    def _build_plugin_algorithm(cfg, kernel=None):
        return _PluginAlgorithm(dict(cfg))

    monkeypatch.setattr(registry_mod, "_ALGORITHMS", None)
    monkeypatch.setattr(registry_mod, "_ALGORITHM_PLUGINS_LOADED", False)
    monkeypatch.setattr(
        registry_mod.metadata,
        "entry_points",
        lambda: _FakeEntryPoints([_FakeEntryPoint("_pytest_plugin_algo", _build_plugin_algorithm)]),
    )

    loaded = registry_mod.discover_algorithm_plugins(force=True)
    registry = registry_mod.get_algorithms_registry()

    assert loaded == ("_pytest_plugin_algo",)
    algorithm = registry["_pytest_plugin_algo"](GenericAlgorithmConfig({"answer": 42}).to_dict(), None)
    result = algorithm.run(problem=None)
    np.testing.assert_allclose(result["F"], np.array([[42.0, 0.0]]))

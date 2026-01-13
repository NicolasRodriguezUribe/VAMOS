from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import GenericAlgorithmConfig
from vamos.engine.algorithm.registry import ALGORITHMS
from vamos.experiment.optimize import OptimizeConfig, optimize_config
from vamos.foundation.problems_registry import ZDT1


@pytest.mark.smoke
def test_unified_optimize_supports_registered_plugin_algorithm() -> None:
    algo_key = "_test_plugin_algo_unified_optimize"

    if algo_key not in ALGORITHMS:

        def _builder(cfg: dict[str, Any], kernel: Any) -> Any:
            class _Algo:
                def run(self, problem: Any, termination: tuple[str, Any], seed: int, eval_strategy=None, live_viz=None):
                    pop_size = int(cfg.get("pop_size", 5))
                    rng = np.random.default_rng(seed)
                    return {
                        "X": np.zeros((pop_size, int(problem.n_var))),
                        "F": rng.random((pop_size, int(problem.n_obj))),
                    }

            return _Algo()

        ALGORITHMS.register(algo_key, _builder)

    result = vamos.optimize("zdt1", algorithm=algo_key, budget=20, pop_size=7, seed=123, verbose=False)
    assert result.F is not None
    assert result.F.shape == (7, 2)


@pytest.mark.smoke
def test_optimize_config_resolves_eval_strategy_string() -> None:
    algo_key = "_test_plugin_algo_eval_strategy_resolution"
    seen: dict[str, Any] = {}

    if algo_key not in ALGORITHMS:

        def _builder(cfg: dict[str, Any], kernel: Any) -> Any:
            class _Algo:
                def run(self, problem: Any, termination: tuple[str, Any], seed: int, eval_strategy=None, live_viz=None):
                    seen["eval_strategy"] = eval_strategy
                    return {
                        "X": np.zeros((2, int(problem.n_var))),
                        "F": np.zeros((2, int(problem.n_obj))),
                    }

            return _Algo()

        ALGORITHMS.register(algo_key, _builder)

    problem = ZDT1(n_var=4)
    cfg = OptimizeConfig(
        problem=problem,
        algorithm=algo_key,
        algorithm_config=GenericAlgorithmConfig({}),
        termination=("n_eval", 5),
        seed=0,
        engine="numpy",
        eval_strategy="serial",
    )
    result = optimize_config(cfg)

    assert result.F is not None
    assert "eval_strategy" in seen
    assert not isinstance(seen["eval_strategy"], str)
    assert seen["eval_strategy"] is not None

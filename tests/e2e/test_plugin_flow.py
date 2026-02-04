"""
E2E Test: Plugin Registration Flow
Verifies the dynamic Registry can accept custom algorithms.
"""

import pytest
import numpy as np
from dataclasses import dataclass

from vamos.engine.algorithm.registry import ALGORITHMS
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos import optimize


@pytest.mark.e2e
def test_plugin_registration_and_usage():
    """
    Simulate external plugin registration:
    1. Define a mock algorithm builder
    2. Register it dynamically
    3. Run optimization using the registered name
    """

    # 1. Define Mock Algorithm Builder
    def mock_algo_builder(cfg, kernel):
        class MockAlgo:
            def run(self, problem, termination, seed, eval_strategy=None, live_viz=None):
                # Return trivial result
                return {
                    "X": np.zeros((5, problem.n_var)),
                    "F": np.random.rand(5, problem.n_obj),
                }

        return MockAlgo()

    # 2. Register (skip if already present from previous run)
    algo_key = "_e2e_mock_algo"
    if algo_key not in ALGORITHMS:
        ALGORITHMS.register(algo_key, mock_algo_builder)

    @dataclass(frozen=True)
    class DummyConfig:
        def to_dict(self) -> dict[str, object]:
            return {}

    # 3. Use
    problem = ZDT1Problem(n_var=4)
    result = optimize(
        problem,
        algorithm=algo_key,
        algorithm_config=DummyConfig(),
        termination=("max_evaluations", 10),
        seed=1,
        engine="numpy",
    )
    assert result.F.shape[0] == 5, "Mock algo should return 5 individuals"

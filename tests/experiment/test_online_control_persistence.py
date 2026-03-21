from __future__ import annotations

from pathlib import Path

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.experiment.observers.storage import StorageObserver
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.observer import RunContext
from vamos.foundation.problem.registry import make_problem_selection


def test_storage_observer_writes_online_control_artifacts(tmp_path: Path) -> None:
    selection = make_problem_selection("zdt1", n_var=4)
    problem = selection.instantiate()
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .selection("tournament", size=2)
        .online_control(enabled=True, policy="adaptive_hierarchical_joint", credit_model="simple_improvement", trace_level="basic")
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("max_evaluations", 12),
        seed=2,
        engine="numpy",
    )

    observer = StorageObserver(output_dir=str(tmp_path))
    observer.on_start(
        RunContext(
            problem=problem,
            algorithm=None,
            config=ExperimentConfig(population_size=6, offspring_population_size=6, max_evaluations=12, seed=2),
            selection=selection,
            algorithm_name="nsgaii",
            engine_name="numpy",
        )
    )
    observer.on_end(
        result.F,
        {
            "algorithm": "nsgaii",
            "engine": "numpy",
            "time_ms": 1.0,
            "evaluations": 12,
            "evals_per_sec": 12.0,
            "spread": 0.1,
            "termination": "max_evaluations",
            "payload": result.data,
            "config": cfg,
            "_kernel_backend": None,
        },
    )

    assert (tmp_path / "online_control_trace.csv").exists()
    assert (tmp_path / "online_control_summary.csv").exists()
    assert (tmp_path / "online_control_run_summary.json").exists()
    assert (tmp_path / "online_control_policy_state.json").exists()

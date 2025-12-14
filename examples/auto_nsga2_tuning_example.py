from __future__ import annotations

import numpy as np

from vamos.foundation.problem.registry import make_problem_selection
from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace
from vamos.engine.tuning.evolver.tuner import NSGAIITuner


def _build_problem(name: str, n_var: int, **kwargs):
    selection = make_problem_selection(name, n_var=n_var)
    return selection.instantiate()


def main():
    # Canonical tuning API: AlgorithmConfigSpace over NSGA-II hyperparameters.
    config_space = AlgorithmConfigSpace.from_template("nsgaii", "default")

    problems = [
        _build_problem("zdt1", 30),
        _build_problem("zdt2", 30),
    ]
    # Reference fronts optional; pass None to compute indicators directly on outputs.
    ref_fronts = [None for _ in problems]

    tuner = NSGAIITuner(
        config_space=config_space,
        problems=problems,
        ref_fronts=ref_fronts,
        indicators=["hv"],
        max_evals_per_problem=3000,
        n_runs_per_problem=1,
        engine="numpy",
        meta_population_size=12,
        meta_max_evals=60,
        seed=42,
    )
    X_meta, F_meta, configs, diagnostics = tuner.optimize()
    best_idx = int(np.argmin(F_meta[:, 0]))
    best_cfg = configs[best_idx]
    print("Best config (decoded):", best_cfg)
    print("Diagnostics:", diagnostics)


if __name__ == "__main__":
    main()

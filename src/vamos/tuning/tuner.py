from __future__ import annotations

import numpy as np

from vamos.tuning.meta_problem import MetaOptimizationProblem
from vamos.tuning.nsga2_meta import MetaNSGAII
from vamos.tuning.parameter_space import AlgorithmConfigSpace


class NSGAIITuner:
    def __init__(
        self,
        config_space: AlgorithmConfigSpace,
        problems,
        ref_fronts,
        indicators,
        max_evals_per_problem: int,
        n_runs_per_problem: int,
        engine: str = "numpy",
        meta_population_size: int = 50,
        meta_max_evals: int = 2000,
        max_total_inner_runs: int | None = None,
        max_wall_time: float | None = None,
        seed: int | None = None,
        optimize_fn=None,
        use_racing: bool = False,
        baseline_quality: float | None = None,
        min_runs_per_problem: int | None = None,
        max_runs_per_problem: int | None = None,
    ):
        self.config_space = config_space
        self.problems = problems
        self.ref_fronts = ref_fronts
        self.indicators = indicators
        self.max_evals_per_problem = max_evals_per_problem
        self.n_runs_per_problem = n_runs_per_problem
        self.engine = engine
        self.meta_population_size = meta_population_size
        self.meta_max_evals = meta_max_evals
        self.max_total_inner_runs = max_total_inner_runs
        self.max_wall_time = max_wall_time
        self.seed = seed
        self.optimize_fn = optimize_fn
        self.use_racing = use_racing
        self.baseline_quality = baseline_quality
        self.min_runs_per_problem = min_runs_per_problem
        self.max_runs_per_problem = max_runs_per_problem

    def optimize(self):
        """
        Run the meta-level NSGA-II and return nondominated meta solutions and decoded configs.
        Returns a tuple (X_meta_nd, F_meta_nd, configs_nd, diagnostics).
        """
        master_rng = np.random.default_rng(self.seed)
        problem_seed = int(master_rng.integers(0, 2**32 - 1))
        algo_seed = int(master_rng.integers(0, 2**32 - 1))
        meta_problem = MetaOptimizationProblem(
            self.config_space,
            self.problems,
            self.ref_fronts,
            self.indicators,
            self.max_evals_per_problem,
            self.n_runs_per_problem,
            self.engine,
            np.random.default_rng(problem_seed),
            optimize_fn=self.optimize_fn,
            min_runs_per_problem=self.min_runs_per_problem,
            max_runs_per_problem=self.max_runs_per_problem,
            use_racing=self.use_racing,
            baseline_quality=self.baseline_quality,
        )
        meta_algo = MetaNSGAII(
            meta_problem,
            population_size=self.meta_population_size,
            offspring_size=self.meta_population_size,
            max_meta_evals=self.meta_max_evals,
            max_total_inner_runs=self.max_total_inner_runs,
            max_wall_time=self.max_wall_time,
            seed=algo_seed,
        )
        X_meta, F_meta, diagnostics = meta_algo.run()
        configs = [self.config_space.decode_vector(vec) for vec in X_meta]
        return X_meta, F_meta, configs, diagnostics

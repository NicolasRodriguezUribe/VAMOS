from __future__ import annotations

import numpy as np

from vamos.algorithm.autonsga2_builder import build_autonsga2
from vamos.metrics.hv_zdt import compute_normalized_hv
from vamos.problem.registry import make_problem_selection
from vamos.tuning.param_space import ParamSpace, Real, Int, Categorical, Condition
from vamos.tuning.random_search_tuner import RandomSearchTuner
from vamos.tuning.tuning_task import TuningTask, Instance, EvalContext


def _build_problem(name: str, n_var: int, **kwargs):
    selection = make_problem_selection(name, n_var=n_var)
    return selection.instantiate()


def main():
    space = ParamSpace(
        params={
            "init.type": Categorical(["random", "lhs", "scatter_search"]),
            "crossover.type": Categorical(["sbx", "blx_alpha"]),
            "crossover.prob": Real(0.6, 1.0),
            "crossover.sbx_eta": Real(5.0, 40.0),
            "crossover.blx_alpha": Real(0.0, 1.0),
            "mutation.type": Categorical(["uniform", "polynomial", "linked_polynomial", "non_uniform"]),
            "mutation.prob_factor": Real(0.1, 2.0),
            "mutation.poly_eta": Real(5.0, 40.0),
            "mutation.uniform_perturb": Real(0.0, 1.0),
            "mutation.non_uniform_perturb": Real(0.0, 1.0),
            "selection.type": Categorical(["tournament", "random"]),
            "selection.tournament_size": Int(2, 6),
            "offspring_size": Int(50, 300),
            "result_mode": Categorical(["population", "external_archive"]),
        },
        conditions=[
            Condition("crossover.sbx_eta", "cfg['crossover.type'] == 'sbx'"),
            Condition("crossover.blx_alpha", "cfg['crossover.type'] == 'blx_alpha'"),
            Condition("mutation.poly_eta", "cfg['mutation.type'] in ['polynomial', 'linked_polynomial']"),
            Condition("selection.tournament_size", "cfg['selection.type'] == 'tournament'"),
            Condition("mutation.uniform_perturb", "cfg['mutation.type'] == 'uniform'"),
            Condition("mutation.non_uniform_perturb", "cfg['mutation.type'] == 'non_uniform'"),
        ],
    )

    instances = [Instance("zdt1", 30), Instance("zdt2", 30), Instance("zdt3", 30)]
    task = TuningTask(
        name="autonsga2_zdt",
        param_space=space,
        instances=instances,
        seeds=[1, 2, 3],
        budget_per_run=20000,
        maximize=True,
        aggregator=np.mean,
    )

    def eval_fn(config, ctx: EvalContext) -> float:
        problem = _build_problem(ctx.instance.name, ctx.instance.n_var, **ctx.instance.kwargs)
        algo = build_autonsga2(config, problem, seed=ctx.seed)
        result = algo.run(problem, termination=("n_eval", ctx.budget), seed=ctx.seed)
        F = result["F"] if isinstance(result, dict) else result.F
        return compute_normalized_hv(F, ctx.instance.name)

    tuner = RandomSearchTuner(task=task, max_trials=10, seed=42)
    best_config, history = tuner.run(eval_fn, verbose=True)
    print("Best config:", best_config)
    print(f"History length: {len(history)}")


if __name__ == "__main__":
    main()

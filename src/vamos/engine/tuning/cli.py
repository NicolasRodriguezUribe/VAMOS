import argparse
import sys
import numpy as np
from typing import Dict, Any

from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.runner import run_single
from vamos.foundation.metrics.hypervolume import compute_hypervolume

from vamos.engine.tuning import (
    RacingTuner,
    Scenario,
    TuningTask,
    Instance,
    EvalContext,
    # Builders
    build_nsgaii_config_space,
    build_moead_config_space,
    build_nsgaiii_config_space,
    build_spea2_config_space,
    build_ibea_config_space,
    build_smpso_config_space,
    build_smsemoa_config_space,
    # Bridge
    config_from_assignment,
)

BUILDERS = {
    "nsgaii": build_nsgaii_config_space,
    "moead": build_moead_config_space,
    "nsgaiii": build_nsgaiii_config_space,
    "spea2": build_spea2_config_space,
    "ibea": build_ibea_config_space,
    "smpso": build_smpso_config_space,
    "smsemoa": build_smsemoa_config_space,
}


def parse_args():
    parser = argparse.ArgumentParser(description="VAMOS Tuning CLI (vamos-tune)")
    parser.add_argument("--problem", type=str, required=True, help="Problem ID (e.g., zdt1)")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=list(BUILDERS.keys()), help="Algorithm to tune")
    parser.add_argument("--n-var", type=int, default=30, help="Number of variables")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives")
    parser.add_argument("--budget", type=int, default=5000, help="Max evaluations per run")
    parser.add_argument("--tune-budget", type=int, default=20000, help="Total tuning budget (evaluations)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--pop-size", type=int, default=100, help="Fixed population size (if not tuning it)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--ref-point", type=str, default=None, help="Reference point for HV (comma-separated, e.g. '1.1,1.1')")
    return parser.parse_args()


def make_evaluator(
    problem_key: str, 
    n_var: int, 
    n_obj: int, 
    algorithm_name: str, 
    fixed_pop_size: int,
    ref_point_str: str | None
):
    """
    Creates an evaluation function that runs the algorithm and returns the Hypervolume.
    """
    # Parse reference point once
    if ref_point_str:
        try:
            ref_point = np.array([float(x.strip()) for x in ref_point_str.split(",")])
            if len(ref_point) != n_obj:
                print(f"Warning: Reference point length ({len(ref_point)}) does not match n_obj ({n_obj}).")
        except ValueError:
            print("Error parsing --ref-point. Using default.")
            ref_point = np.array([1.1] * n_obj)
    else:
        # Default fallback
        ref_point = np.array([1.1] * n_obj)

    def eval_fn(config_dict: Dict[str, Any], ctx: EvalContext) -> float:
        try:
            # 1. Prepare Configuration
            # Merge fixed params if missing in tuning config
            start_config = dict(config_dict)
            if "pop_size" not in start_config:
                 start_config["pop_size"] = fixed_pop_size
            
            # Use the unified bridge to build the AlgorithmConfig object
            cfg = config_from_assignment(algorithm_name, start_config)
            
            # 2. Run the algorithm
            selection = make_problem_selection(problem_key, n_var=n_var, n_obj=n_obj)
            
            exp_cfg = ExperimentConfig(
                population_size=cfg.population_size, 
                max_evaluations=ctx.budget, 
                seed=ctx.seed
            )
            
            result = run_single(
                engine="numpy", # Defaulting to numpy for stability in tuning
                algorithm=algorithm_name,
                problem_selection=selection,
                experiment_config=exp_cfg,
                algorithm_config=cfg,
                verbose=False
            )
            
            # 3. Compute metric (Hypervolume)
            if result.F is None or len(result.F) == 0:
                return 0.0
                
            hv = compute_hypervolume(result.F, ref_point)
            return hv
            
        except Exception as e:
            # In tuning, we often want to absorb errors and return bad score
            # to keep the racer alive.
            print(f"Eval failed for {algorithm_name}: {e}")
            return 0.0

    return eval_fn


def main():
    args = parse_args()
    
    # 1. Define Parameter Space
    builder = BUILDERS.get(args.algorithm)
    if not builder:
        raise ValueError(f"Unknown algorithm {args.algorithm}")
    
    param_space = builder()

    print(f"Tuning {args.algorithm} on {args.problem} (Budget: {args.tune_budget})")
    print(f"Parallel Jobs: {args.n_jobs}")
    
    # 2. Setup Scenario and Task
    scenario = Scenario(
        max_experiments=args.tune_budget,
        initial_budget_per_run=args.budget,
        use_adaptive_budget=False,
        verbose=True,
        n_jobs=args.n_jobs
    )
    
    # Instance definition
    instances = [Instance(name=args.problem, n_var=args.n_var, kwargs={})]
    
    # Seeds
    seeds = [args.seed + i for i in range(100)]
    
    # Aggregator: Mean
    aggregator = lambda scores: np.mean(scores)
    
    task = TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=aggregator,
        budget_per_run=args.budget,
        maximize=True, # HV is maximization
    )
    
    # 3. Run Tuner
    tuner = RacingTuner(task=task, scenario=scenario, seed=args.seed)
    
    eval_fn = make_evaluator(
        args.problem, 
        args.n_var, 
        args.n_obj, 
        args.algorithm, 
        args.pop_size,
        args.ref_point
    )
    
    best_config, history = tuner.run(eval_fn)
    
    print("\n--- Tuning Complete ---")
    print("Best Configuration Found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    print("\nUse these parameters in your scripts or CLI using --config!")

if __name__ == "__main__":
    main()

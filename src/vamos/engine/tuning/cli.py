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
    build_nsgaii_config_space,
    build_moead_config_space,
    config_from_assignment,
)
from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig


def parse_args():
    parser = argparse.ArgumentParser(description="VAMOS Tuning CLI (vamos-tune)")
    parser.add_argument("--problem", type=str, required=True, help="Problem ID (e.g., zdt1)")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=["nsgaii", "moead"], help="Algorithm to tune")
    parser.add_argument("--n-var", type=int, default=30, help="Number of variables")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives")
    parser.add_argument("--budget", type=int, default=5000, help="Max evaluations per run")
    parser.add_argument("--tune-budget", type=int, default=20000, help="Total tuning budget (evaluations)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--pop-size", type=int, default=100, help="Fixed population size (if not tuning it)")
    return parser.parse_args()


def make_evaluator(problem_key: str, n_var: int, n_obj: int, algorithm_name: str, fixed_pop_size: int):
    """
    Creates an evaluation function that runs the algorithm and returns the Hypervolume.
    """
    # Pre-select problem to avoid overhead in the loop, though resolving happens inside run_single usually
    # But here we just capture the specs
    
    def eval_fn(config_dict: Dict[str, Any], ctx: EvalContext) -> float:
        # 1. Convert flat config dict to Algorithm Config object
        # Note: The tuning module's spaces usually map to flat dicts. We need to reconstruct the Config object.
        # For simplicity in this CLI, we assume the config space builder keys match the Config expects or we map them.
        
        # We need a robust way to map the tuned parameters to the config class
        # This depends on how build_nsgaii_config_space sets up parameter names.
        # Assuming they are prefixed or match arguments.
        
        try:
             # Basic mapping for NSGA-II:
            if algorithm_name == "nsgaii":
                # We start with a default config
                cfg = NSGAIIConfig().engine("numpy") # Use numpy for tuning speed/reliability default
                
                # Update with tuned values
                # Example: if config_dict has 'crossover.prob', we set it.
                # The generic way is using config_from_assignment if available or manual mapping
                # checking what build_nsgaii_config_space produces.
                # For now, let's assume a direct mapping or simple manual one for common params
                
                 # Manual mapping for common AutoNSGA-II params if they exist in config_dict
                if "pop_size" in config_dict:
                    cfg.pop_size(int(config_dict["pop_size"]))
                else:
                    cfg.pop_size(fixed_pop_size)
                    
                if "crossover_prob" in config_dict:
                    cfg.crossover("sbx", prob=config_dict["crossover_prob"], eta=config_dict.get("crossover_eta", 20.0))
                
                if "mutation_prob" in config_dict:
                     cfg.mutation("pm", prob=config_dict["mutation_prob"], eta=config_dict.get("mutation_eta", 20.0))
                     
            elif algorithm_name == "moead":
                 cfg = MOEADConfig().engine("numpy")
                 if "pop_size" in config_dict:
                    cfg.pop_size(int(config_dict["pop_size"]))
                 else:
                    cfg.pop_size(fixed_pop_size)
            
            # 2. Run the algorithm
            # We use a fresh seed for the run from the context
            run_seed = ctx.seed
            
            selection = make_problem_selection(problem_key, n_var=n_var, n_obj=n_obj)
            
            # Prepare ExperimentConfig just for the run limits
            exp_cfg = ExperimentConfig(
                population_size=cfg.population_size, 
                max_evaluations=ctx.budget, 
                seed=run_seed
            )
            
            result = run_single(
                engine="numpy",
                algorithm=algorithm_name,
                problem_selection=selection,
                experiment_config=exp_cfg,
                algorithm_config=cfg,
                verbose=False
            )
            
            # 3. Compute metric (Hypervolume)
            # We need a reference point. For ZDT1 (2 objectives), typical is [1.1, 1.1] or similar
            # Ideally this comes from the problem registry or we approximate it.
            # For this MVP CLI, we use a fixed ref point common for normalized ZDT/DTLZ
            ref_point = np.array([1.1] * n_obj)
            
            if result.F is None or len(result.F) == 0:
                return 0.0
                
            hv = compute_hypervolume(result.F, ref_point)
            return hv
            
        except Exception as e:
            print(f"Eval failed: {e}")
            return 0.0

    return eval_fn


def main():
    args = parse_args()
    
    # 1. Define Parameter Space
    if args.algorithm == "nsgaii":
        param_space = build_nsgaii_config_space() 
        # Note: build_nsgaii_config_space might need to be checked for what it returns.
        # If it's empty or basic, we might need to add params here manually if the builder is simple.
        # Assuming it returns a populated ParamSpace.
    elif args.algorithm == "moead":
        param_space = build_moead_config_space()
    else:
        raise ValueError(f"Unknown algorithm {args.algorithm}")

    print(f"Tuning {args.algorithm} on {args.problem} (Budget: {args.tune_budget})")
    
    # 2. Setup Scenario and Task
    scenario = Scenario(
        max_experiments=args.tune_budget,
        initial_budget_per_run=args.budget, # Fixed budget per run for now, essentially
        use_adaptive_budget=False, # Simplify for MVP
        verbose=True
    )
    
    # We treat the problem as a single instance for now, but tuning usually generalizes over instances.
    # Here we define one "instance" which is just the problem key
    instances = [Instance(name=args.problem, n_var=args.n_var, kwargs={})]
    
    # Seeds for the tuner to use for runs
    seeds = [args.seed + i for i in range(100)] # ample seeds
    
    # Aggregator: Mean hypervolume
    aggregator = lambda scores: np.mean(scores)
    
    task = TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=aggregator,
        budget_per_run=args.budget,
        maximize=True, # Hypervolume is maximization
    )
    
    # 3. Run Tuner
    tuner = RacingTuner(task=task, scenario=scenario, seed=args.seed)
    
    eval_fn = make_evaluator(
        args.problem, 
        args.n_var, 
        args.n_obj, 
        args.algorithm, 
        args.pop_size
    )
    
    best_config, history = tuner.run(eval_fn)
    
    print("\n--- Tuning Complete ---")
    print("Best Configuration Found:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    # Optional: rerun best
    print("\nUse these parameters in your scripts or CLI using --config!")

if __name__ == "__main__":
    main()

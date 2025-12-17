---
applyTo: "examples/**/*.py"
description: Example script guidelines for VAMOS
---

# Example Script Standards

## Purpose
Examples in `examples/` demonstrate VAMOS features with runnable, self-contained scripts.

## Structure
```python
"""
One-line description of what this example demonstrates.

Usage:
    python examples/this_example.py

Requirements:
    pip install -e ".[examples]"  # if extra deps needed
"""
from __future__ import annotations

# Use the public API facades for user-friendly imports
from vamos import (
    NSGAIIConfig,
    OptimizeConfig,
    optimize,
    make_problem_selection,
    plot_pareto_front_2d,
)
# For real-world problems:
# from vamos import FeatureSelectionProblem, HyperparameterTuningProblem, WeldedBeamDesignProblem

def main():
    # 1. Setup problem
    selection = make_problem_selection("zdt1", n_var=30)
    problem = selection.instantiate()
    
    # 2. Configure algorithm
    cfg = (
        NSGAIIConfig()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .fixed()
    )
    
    # 3. Run optimization
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 10000),
            seed=42,
        )
    )
    
    # 4. Output/visualize
    print(f"Found {result.F.shape[0]} solutions")

if __name__ == "__main__":
    main()
```

## Guidelines
- **Self-contained**: No dependencies on other examples
- **Runnable**: `python examples/example.py` should work after install
- **Documented**: Module docstring with usage and requirements
- **Seeded**: Always set `seed=` for reproducibility
- **Reasonable budget**: 5000-25000 evaluations (runs in < 1 min)

## Naming
- `{feature}_example.py` for feature demos
- `{problem}_{algorithm}.py` for specific setups
- `{use_case}_pipeline.py` for multi-step workflows

## Imports - Use Public API
Prefer the user-friendly public API over deep internal imports:
```python
# GOOD - User-friendly public API
from vamos import optimize, OptimizeConfig, NSGAIIConfig
from vamos import ZDT1, make_problem_selection
from vamos import plot_pareto_front_2d, plot_pareto_front_3d
from vamos import FeatureSelectionProblem, HyperparameterTuningProblem

# AVOID - Deep internal imports (for advanced/extension work only)
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.core.optimize import optimize
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem
```

## Available Public API Imports
```python
# Core optimization
from vamos import optimize, OptimizeConfig, OptimizationResult, ExperimentConfig

# Algorithms and configs
from vamos import (
    NSGAII, NSGAIIConfig,
    MOEAD, MOEADConfig,
    SMSEMOA, SMSEMOAConfig,
    SPEA2, SPEA2Config,
    IBEA, IBEAConfig,
    SMPSO, SMPSOConfig,
    NSGAIII, NSGAIIIConfig,
)

# Benchmark problems
from vamos import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from vamos import DTLZ1, DTLZ2, DTLZ3, DTLZ4
from vamos import WFG1
from vamos import make_problem_selection, available_problem_names

# Real-world problems
from vamos import FeatureSelectionProblem, HyperparameterTuningProblem, WeldedBeamDesignProblem

# Tuning
from vamos import ParamSpace, RandomSearchTuner, RacingTuner

# Visualization
from vamos import plot_pareto_front_2d, plot_pareto_front_3d, plot_hv_convergence

# MCDM / decision-making
from vamos import weighted_sum_scores, tchebycheff_scores, knee_point_scores

# Statistics
from vamos import friedman_test, pairwise_wilcoxon, plot_critical_distance
```

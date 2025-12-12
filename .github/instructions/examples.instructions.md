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

# Standard setup
from vamos.problem.registry import make_problem_selection
from vamos.core.runner import run_single, ExperimentConfig
# ... other imports

def main():
    # 1. Setup problem
    selection = make_problem_selection("zdt1", n_var=30)
    
    # 2. Configure experiment
    cfg = ExperimentConfig(
        population_size=100,
        max_evaluations=10000,
        seed=42,
    )
    
    # 3. Run
    result = run_single("numpy", "nsgaii", selection, cfg)
    
    # 4. Output/visualize
    print(f"Found {result['F'].shape[0]} solutions")

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

## Imports
Use installed package, not relative:
```python
# Good
from vamos.problem.registry import make_problem_selection

# Bad
import sys; sys.path.insert(0, "../src")
```

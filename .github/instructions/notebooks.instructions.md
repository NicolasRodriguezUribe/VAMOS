---
applyTo: "notebooks/**/*.ipynb"
description: Jupyter notebook guidelines for VAMOS
---

# Notebook Standards for VAMOS

## Purpose
Notebooks in `notebooks/` are for **exploration and demonstration**, not production logic.
If code becomes reusable, promote it to `src/vamos/` and add tests.

## Structure
1. **Title cell** — Markdown with notebook purpose
2. **Setup cell** — Imports and path setup for editable install
3. **Problem setup** — Use `make_problem_selection()`
4. **Algorithm config** — Use builder pattern, not raw dicts
5. **Run and visualize** — Keep runs short for reproducibility

## Standard Setup Cell
```python
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src is importable for editable installs
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if (ROOT / "src").exists() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

# User-friendly public API
from vamos import (
    optimize, OptimizeConfig, NSGAIIConfig,
    make_problem_selection,
    plot_pareto_front_2d,
)
```

## Visualization
- Use `plot_pareto_front_2d()` from `vamos` for 2D fronts
- Keep figures self-contained with titles and labels
- Save figures to `results/` if needed for reports

## Budgets
Keep evaluation budgets small for fast iteration:
- Smoke/demo: `max_evaluations=1000`
- Full run: `max_evaluations=10000-50000`

## Seeds
Always set seeds for reproducibility:
```python
from vamos import OptimizeConfig, NSGAIIConfig

cfg = (NSGAIIConfig()
    .pop_size(100)
    .crossover("sbx", prob=0.9, eta=20.0)
    .mutation("pm", prob="1/n", eta=20.0)
    .selection("tournament", pressure=2)
    .survival("nsga2")
    .engine("numpy")
    .fixed())
result = optimize(
    OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 5000),
        seed=42,
    )
)
```

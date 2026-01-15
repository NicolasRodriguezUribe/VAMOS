# VAMOS: Vectorized Architecture for Multiobjective Optimization Studies

> **A high-performance, unified framework for Multi-Objective Evolutionary Algorithms (MOEA) in Python.**

![VAMOS Banner](docs/assets/VAMOS.jpeg)

VAMOS bridges the gap between simple research scripts and large-scale optimization studies. It provides a unified API for running state-of-the-art algorithms across diverse problems, backed by vectorized kernels (NumPy, Numba, JAX) for maximum performance.

## 🚀 Key Features

- **Unified API**: A clear, fluent interface `vamos.optimize()` for all workflows.
- **Battle-Tested Algorithms**: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA.
- **Unified Archiving**: Consistent external archive configuration `.archive(size, type="epsilon_grid")` across all algorithms.
- **Multi-Fidelity Tuning**: Hyperband-style racing with warm-start checkpoints for sample-efficient algorithm configuration.
- **Performance Driven**: Vectorized kernels, GPU acceleration (JAX), and optional Numba JIT compilation.
- **Interactive Analysis**: Built-in dashboards with `explore_result_front(result)` and publication-ready LaTeX tables.
- **Extensible**: Standardized protocols for adding custom problems, operators, and algorithms.

## 📦 Quick Install

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core + essential extras
pip install -e ".[compute,research,analysis]"
```

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core + essential extras
pip install -e ".[compute,research,analysis]"
```

## ⚡ Quickstart

Solve the ZDT1 benchmark problem with NSGA-II in just a few lines:

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    budget=10000,
    pop_size=100,
    engine="numpy",
    seed=42,
)

front = result.front()
print(f"Non-dominated solutions: {len(front) if front is not None else 0}")
# from vamos.ux.api import plot_result_front
# plot_result_front(result)  # Quick Pareto front plot
```

Prefer a guided CLI? Run:

```bash
vamos quickstart
```

This wizard writes a reusable config and stores results under `results/quickstart/`.

Use `vamos quickstart --template list` to see domain templates.

New to Python? Start with the Minimal Python Track: `docs/guide/minimal-python.md`.

After a run, summarize results with:

```bash
vamos summarize --results results/quickstart
```

New CLI helpers:

- `vamos quickstart`: guided wizard that writes a config
- `vamos summarize`: table/JSON summary of recent runs
- `vamos open-results`: print or open the latest run folder

Preferred path: start with `optimize(...)`. Use config objects only when you need fully specified, reproducible runs or plugin algorithms.
See `docs/guide/getting-started.md` for a short decision guide.

Advanced path (explicit config objects):

```python
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=30)
algo = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)

result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo,
    termination=("n_eval", 10000),
    seed=42,
    engine="numpy",
)
```

Reminder: plain dict configs are intentionally not accepted (use `GenericAlgorithmConfig` for plugin algorithms).

## Notes

- NSGA-III works best when `pop_size` matches the number of reference directions. With `reference_directions(divisions=p)`, the count is `comb(p + n_obj - 1, n_obj - 1)`; mismatches emit a warning unless strict enforcement is enabled.
- RVEA requires `pop_size` to match the number of reference directions implied by `n_partitions` (simplex-lattice). It uses APD survival with periodic reference-vector adaptation (`alpha`, `adapt_freq`).
- For reproducible results, set `seed`; NumPy/Numba/MooCore backends share the same RNG-driven stochastic operators.
- Default operator settings align with jMetalPy standard configurations (e.g., SBX prob 1.0, PM prob 1/n, MOEA/D PBI, IBEA kappa 1.0); override via config/CLI if needed.
- Troubleshooting guide: `docs/guide/troubleshooting.md`.

## 📚 Examples & Notebooks

VAMOS comes with a comprehensive suite of Jupyter notebooks organized by tier:

- **0. Basic**: Essential concepts and API basics.
  - `notebooks/0_basic/01_quickstart.ipynb`
- **1. Intermediate**: Real-world problems, constraints, and deeper analysis.
  - `notebooks/1_intermediate/10_discrete_problems.ipynb`
  - `notebooks/1_intermediate/16_interactive_explorer.ipynb`
- **2. Advanced**: Custom extensions, distributed evaluation, and research benchmarks.
  - `notebooks/2_advanced/30_paper_benchmarking.ipynb`

## 🛠️ Tooling Ecosystem

- **`vamos-profile`**: Analyze the performance overhead of your experiments.
  ```bash
  vamos-profile nsgaii zdt1 --budget 5000
  ```
- **`vamos-benchmark`**: Generate full reports comparing multiple algorithms, plus jMetalPy-compatible lab outputs (`summary/lab/QualityIndicatorSummary.csv`, Wilcoxon tables, boxplots). Boxplots require `matplotlib`.
  ```bash
  vamos-benchmark --suite ZDT_small --algorithms nsgaii moead --output report/
  ```
- **`vamos-tune`**: Racing-style hyperparameter tuning with optional multi-fidelity. `--tune-budget` counts configuration evaluations; `--budget` is per-run evaluations.
  ```bash
  vamos-tune --problem zdt1 --algorithm nsgaii --budget 5000 --tune-budget 200 --n-seeds 5
  ```
- **`vamos-self-check`**: Verify your installation and backend availability.

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

- **Found a bug?** Open an issue.
- **Want to add an algorithm?** Check `dev/add_algorithm.md` in the docs.
- **Using AI tools?** Read `.agent/docs/AGENTS.md` for our AI coding standards.

---

**VAMOS** is a research-oriented multi-objective optimization framework.

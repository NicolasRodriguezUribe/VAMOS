# VAMOS: Vectorized Architecture for Multiobjective Optimization Studies

> **A high-performance, unified framework for Multi-Objective Evolutionary Algorithms (MOEA) in Python.**

![VAMOS Banner](docs/assets/VAMOS.jpeg)

VAMOS bridges the gap between simple research scripts and large-scale optimization studies. It provides a unified API for running state-of-the-art algorithms across diverse problems, backed by vectorized kernels (NumPy, Numba, JAX) for maximum performance.

## 🚀 Key Features

- **Unified API**: A clear, fluent interface `vamos.optimize()` for all workflows.
- **Battle-Tested Algorithms**: NSGA-II/III, MOEA/D, SMS-EMOA, SPEA2, IBEA, SMPSO, AGE-MOEA, RVEA.
- **Unified Archiving**: Consistent external archive configuration via `.archive(size=...).archive_type("crowding" | "hypervolume")`.
- **Multi-Fidelity Tuning**: Hyperband-style racing with warm-start checkpoints for sample-efficient algorithm configuration.
- **Ready-to-use Tuning Backends**: `racing` and `random` work out of the box; install the optional `tuning` extra to enable `optuna`, `bohb_optuna`, `smac3`, and `bohb` via `vamos tune`.
- **Performance Driven**: Vectorized kernels, GPU acceleration (JAX), and optional Numba JIT compilation.
- **Interactive Analysis**: Built-in dashboards with `explore_result_front(result)` and publication-ready LaTeX tables.
- **Visual Problem Builder**: Define custom problems in the browser with live Pareto front preview via VAMOS Studio.
- **Extensible**: Standardized protocols for adding custom problems, operators, and algorithms.

## 📦 Quick Install

```bash
pip install vamos
```

For development and extras:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core + essential extras
pip install "vamos[compute,research,analysis]"
```

Optional model-based tuning backends (`optuna`, `smac3`, `bohb`):

```bash
pip install "vamos[tuning]"
```

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core + essential extras
pip install "vamos[compute,research,analysis]"
```

## ⚡ Quickstart

Solve the ZDT1 benchmark problem with NSGA-II in just a few lines:

```python
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=10000,
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

All functionality lives under one command. Run `vamos help` to list everything:

| Command | What it does |
|---------|-------------|
| `vamos quickstart` | Guided wizard that writes a config |
| `vamos create-problem` | Scaffold a custom problem file |
| `vamos summarize` | Table/JSON summary of recent runs |
| `vamos check` | Verify installation and backends |
| `vamos bench` | Benchmark suite across algorithms |
| `vamos studio` | Launch interactive dashboard |
| `vamos tune` | Hyperparameter tuning |
| `vamos profile` | Performance profiling |
| `vamos zoo` | Problem zoo presets |

## 🎯 Tuning Quick Start

You can use the implemented tuning backends directly from `vamos tune`:
`racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, `bohb`.

Check backend availability in your current environment:

```bash
vamos tune --list-backends
```

Note: `racing` and `random` require no extra dependencies. The model-based backends (`optuna`, `bohb_optuna`, `smac3`, `bohb`) require the optional `tuning` extra: `pip install "vamos[tuning]"`.

Recommended robust command (fallback + suite-stratified split):

```bash
vamos tune \
  --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 \
  --algorithm nsgaii \
  --backend optuna \
  --backend-fallback random \
  --split-strategy suite_stratified \
  --budget 5000 \
  --tune-budget 200 \
  --n-jobs -1
```

Canonical tuning reference: `docs/topics/tuning.md`.

New to hands-on learning? Open the **interactive tutorial notebook**:

```bash
jupyter notebook notebooks/0_basic/05_interactive_tutorial.ipynb
```

## 🧩 Define Your Own Problem

Use `make_problem()` to turn any Python function into a VAMOS-compatible problem
-- no classes, no protocols, no NumPy vectorization required:

```python
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
)

result = optimize(problem, algorithm="nsgaii", max_evaluations=5000, seed=42)
```

Your function receives a single solution `x` (array of length `n_var`) and returns
a list of `n_obj` objective values. VAMOS auto-vectorizes it for performance.

For extra speed, pass `vectorized=True` and write a function that handles batches directly.

Prefer a file template? The CLI wizard scaffolds a ready-to-run `.py` file:

```bash
vamos create-problem
# Prompts for: name, variables, objectives, bounds, style
# Generates a .py file with TODO markers -- fill in your math and run it
```

Or use the **visual builder** in VAMOS Studio -- write your objectives
in the browser, pick an algorithm, and see the Pareto front update on each run:

```bash
vamos studio
# Open the "Problem Builder" tab
```

See `docs/dev/add_problem.md` for all approaches (function, class, or registry).

## VAMOS Assist (no-code workflow)

VAMOS Assist provides an end-to-end no-code flow for creating validated experiment plans, materializing runnable projects, and optionally running smoke checks. You can start with deterministic templates (no API keys), and optionally use provider-backed auto planning.

```bash
vamos assist go "template-first example" --template demo --smoke
```

```bash
pip install vamos[openai]
setx OPENAI_API_KEY "..."
vamos assist go "..." --mode auto --provider openai --smoke
```

See `docs/assist.md` for the full guide (billing, privacy, artifacts, troubleshooting).

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
    termination=("max_evaluations", 10000),
    seed=42,
    engine="numpy",
)
```

Reminder: plain dict configs are intentionally not accepted (use `GenericAlgorithmConfig` for plugin algorithms).

## Notes

- For reproducible results, set `seed`; NumPy/Numba/MooCore backends share the same RNG-driven stochastic operators.
- Troubleshooting guide: `docs/guide/troubleshooting.md`.
- Algorithm-specific notes (reference directions, operator defaults): `docs/reference/algorithms.md`.
- Release packaging smoke checklist: `docs/release_smoke.md`.

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

All tools are available as `vamos <subcommand>`. Run `vamos help` for the full list.

- **`vamos profile`**: Analyze the performance overhead of your experiments.
  ```bash
  vamos profile nsgaii zdt1 --budget 5000
  ```
- **`vamos bench`**: Generate full reports comparing multiple algorithms, plus jMetalPy-compatible lab outputs (`summary/lab/QualityIndicatorSummary.csv`, Wilcoxon tables, boxplots). Boxplots require `matplotlib`.
  ```bash
  vamos bench --suite ZDT_small --algorithms nsgaii moead --output report/
  ```
- **`vamos tune`**: You can use the implemented tuners directly from CLI (`racing`, `random`, `optuna`, `bohb_optuna`, `smac3`, `bohb`). `--tune-budget` counts configuration evaluations; `--budget` is per-run evaluations.
  ```bash
  vamos tune --problem zdt1 --algorithm nsgaii --budget 5000 --tune-budget 200 --n-seeds 5
  ```
- Recommended robust invocation (backend fallback + suite-stratified split):
  ```bash
  vamos tune --instances zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1 --algorithm nsgaii --backend optuna --backend-fallback random --split-strategy suite_stratified --budget 5000 --tune-budget 200 --n-jobs -1
  ```
- Full tuning reference (canonical docs): `docs/topics/tuning.md`.
- Generic tuning example (script-based):
  ```bash
  python examples/tuning/racing_tuner_generic.py --algorithm nsgaii --multi-fidelity --fidelity-levels 500,1000,1500
  ```
- **`vamos check`**: Verify your installation and backend availability.

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

- **Found a bug?** Open an issue.
- **Want to add an algorithm?** Check `dev/add_algorithm.md` in the docs.
- **Using AI tools?** Read `.agent/docs/AGENTS.md` for our AI coding standards.
- **Troubleshooting**: `docs/guide/troubleshooting.md`.
- **Security issues**: See `SECURITY.md` for private reporting.
- **Contributors**: See `AUTHORS.md`.

---

**VAMOS** is a research-oriented multi-objective optimization framework.

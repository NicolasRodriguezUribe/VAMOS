# Minimal Python Track

If you can run a command and copy-paste, this track is for you.
Goal: run an experiment and get results in minutes, without learning all the jargon up front.

## 1. Install

Create a virtual environment and install VAMOS:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[analysis]"
```

If you do not need plots, you can install just the core:

```bash
pip install -e .
```

## 2. Run the guided wizard

The quickstart wizard asks a few questions, writes a config file, and runs one experiment:

```bash
vamos quickstart
```

Want to see domain-flavored templates?

```bash
vamos quickstart --template list
```

Run a template without prompts:

```bash
vamos quickstart --template physics_design --yes --no-plot
```

## 3. Find your results

Results are stored under:

```
results/quickstart/<PROBLEM>/.../seed_<N>/
```

Key files:
- `FUN.csv`: objective values (your solutions)
- `metadata.json`: run details
- `resolved_config.json`: the final settings that were used
- `pareto_front_*.png`: plot (if enabled)

## 4. Re-run or change settings

Re-run the same config:

```bash
vamos --config results/quickstart/quickstart_YYYYMMDD_HHMMSS.json
```

Change budget or seed without editing files:

```bash
vamos --config results/quickstart/quickstart_YYYYMMDD_HHMMSS.json --max-evaluations 8000 --seed 1
```

## 5. Quick summary

List recent runs:

```bash
vamos summarize --results results/quickstart
```

Open the latest run folder:

```bash
vamos open-results --results results/quickstart --open
```

## Glossary (plain language)

- Problem: the task you want to optimize (a dataset or a math function).
- Algorithm: the search method (default is NSGA-II).
- Objective: the quantity you want to minimize (often two or more).
- Pareto front: the best trade-offs found so far (no single solution is best at everything).
- Budget: how many evaluations to spend (more budget = longer run).
- Population size: how many candidate solutions are kept each step.
- Seed: fixes randomness so runs are repeatable.
- Engine: compute backend (numpy is the default).

## If something fails

- Run: `vamos-self-check` to verify your install.
- For biology/chemistry templates, install scikit-learn:
  `pip install -e ".[examples]"`.
- For plots, install the analysis extras:
  `pip install -e ".[analysis]"`.

## Next steps

- `docs/guide/getting-started.md` for the full API overview.
- `docs/guide/cli.md` for CLI details and config files.
- `docs/guide/cookbook.md` for copy-paste recipes.

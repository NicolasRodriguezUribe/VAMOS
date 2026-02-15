# CLI Module

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates (layer/monolith/public-api/import/optional-deps/logging/no-print/no-shims).
- ADRs in `docs/dev/adr/` are mandatory reading before architectural changes.


This directory contains the command-line interface for VAMOS.

## Entry Point

```bash
python -m vamos.experiment.cli.main [OPTIONS]
# or after install:
vamos [OPTIONS]
```

## Subcommands

All VAMOS functionality is available via `vamos <subcommand>`. Run `vamos help` to list them.

| Command | Description | File |
|---------|-------------|------|
| `vamos quickstart` | Guided wizard for a single run | `quickstart.py` |
| `vamos create-problem` | Scaffold a custom problem file | `create_problem.py` |
| `vamos summarize` | Table/JSON summary of results | `results_cli.py` |
| `vamos open-results` | Print or open the latest run folder | `results_cli.py` |
| `vamos ablation` | Run ablation studies | `ablation.py` |
| `vamos assist` | AI-assisted experiment planning | `../../assist/cli.py` |
| `vamos check` | Verify installation and backends | `../diagnostics/self_check.py` |
| `vamos bench` | Benchmark suite across algorithms | `../benchmark/cli.py` |
| `vamos studio` | Launch interactive dashboard | `../../ux/studio/app.py` |
| `vamos zoo` | Problem zoo presets | `../zoo/cli.py` |
| `vamos tune` | Hyperparameter tuning | `tune.py` |
| `vamos profile` | Performance profiling | `../profiler/cli.py` |

> All tools are accessed via `vamos <subcommand>`. Legacy `vamos-*` entry points have been removed.

## Key Options (standard run)

| Flag | Description | Default |
|------|-------------|---------|
| `--problem` | Problem name (zdt1, dtlz2, etc.) | zdt1 |
| `--algorithm` | Algorithm (nsgaii, moead, spea2, etc.) | nsgaii |
| `--engine` | Backend (numpy, numba, moocore, jax) | numpy |
| `--max-evaluations` | Termination budget | 25000 |
| `--population-size` | Population size | 100 |
| `--n-var` | Number of variables | problem default |
| `--n-obj` | Number of objectives | problem default |
| `--config` | YAML config file path | None |
| `--seed` | Random seed | 42 |
| `--output-root` | Results directory | results/ |

## `create-problem` subcommand

Generates a ready-to-run `.py` file with a custom problem template.

```bash
# Interactive wizard
vamos create-problem

# Non-interactive with explicit args
vamos create-problem --name "my problem" --n-var 5 --n-obj 3 --yes

# Class-based template instead of functional
vamos create-problem --style class --output my_problem.py
```

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Problem name | my_problem |
| `--n-var` | Number of decision variables | 2 |
| `--n-obj` | Number of objectives | 2 |
| `--output`, `-o` | Output file path | `<name>.py` |
| `--style` | `functional` (uses `make_problem`) or `class` | functional |
| `--budget` | Max evaluations in generated script | 5000 |
| `--yes` | Accept defaults without prompting | false |

## Config File Override

CLI flags override YAML config values:
```bash
python -m vamos.experiment.cli.main --config study.yaml --max-evaluations 5000
```

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Entry point, subcommand dispatch |
| `create_problem.py` | `create-problem` wizard and template generation |
| `quickstart.py` | `quickstart` wizard |
| `results_cli.py` | `summarize` and `open-results` commands |
| `ablation.py` | `ablation` command |
| `parser.py` | Argument parsing for standard runs |
| `validation.py` | CLI argument validation |

## Adding CLI Options

1. Add argument to `argparse` in `main.py`
2. Wire to `run_single()` or `ExperimentConfig`
3. Document in `docs/guide/cli.md`
4. Add integration test in `tests/test_cli_*.py`

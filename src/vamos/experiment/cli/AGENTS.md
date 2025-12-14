# CLI Module

This directory contains the command-line interface for VAMOS.

## Entry Point

```bash
python -m vamos.experiment.cli.main [OPTIONS]
# or after install:
vamos [OPTIONS]
```

## Key Options

| Flag | Description | Default |
|------|-------------|---------|
| `--problem` | Problem name (zdt1, dtlz2, etc.) | zdt1 |
| `--algorithm` | Algorithm (nsgaii, moead, spea2, etc.) | nsgaii |
| `--engine` | Backend (numpy, numba, moocore) | numpy |
| `--max-evaluations` | Termination budget | 25000 |
| `--population-size` | Population size | 100 |
| `--n-var` | Number of variables | problem default |
| `--n-obj` | Number of objectives | problem default |
| `--config` | YAML config file path | None |
| `--seed` | Random seed | None |
| `--output-dir` | Results directory | results/ |

## Config File Override

CLI flags override YAML config values:
```bash
python -m vamos.experiment.cli.main --config study.yaml --max-evaluations 5000
```

## Architecture

| File | Purpose |
|------|---------|
| `main.py` | Entry point, argument parsing |
| `commands/` | Subcommand implementations (if present) |

## Adding CLI Options

1. Add argument to `argparse` in `main.py`
2. Wire to `run_single()` or `ExperimentConfig`
3. Document in `docs/cli.md`
4. Add integration test in `tests/test_cli_*.py`

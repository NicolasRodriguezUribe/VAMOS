# Ablation Planning Package

## Architecture Health (must-read)
- Follow `docs/dev/architecture_health.md` before adding new modules, APIs, or dependencies.
- PRs must pass the health gates.

## Overview

This package defines **ablation planning utilities** for tuning experiments.
It builds reproducible grids of (problem × variant × seed) tasks and keeps the
engine layer free of experiment orchestration.

## Structure

- `types.py`: `AblationVariant`, `AblationTask`, `AblationPlan`
- `plan.py`: `build_ablation_plan` task matrix builder
- `__init__.py`: public exports

## Conventions

- No experiment-layer imports or `optimize()` calls here.
- Variants are declarative: use `config_overrides` and `variant.apply(...)`.
- Keep import-time purity (no side effects).

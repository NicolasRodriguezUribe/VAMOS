"""
fANOVA validation: verify that parameter role taxonomy aligns with empirical
importance measured via functional ANOVA on random NSGA-II configurations.

Procedure:
  1. Sample N random NSGA-II configs on a given problem (e.g. DTLZ2-10d).
  2. Evaluate each config → HV score.
  3. Fit fANOVA (via SMAC's fanova or a simple forest-based importance).
  4. Group per-parameter importances by role.
  5. Print summary and save CSV.

Expected result: structural and operator parameters dominate importance in
the first phases; operator_rate and adaptive are fine-tuning knobs.

Usage:
  python experiments/fanova_validation.py [--n_configs 500] [--problem dtlz2] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

N_JOBS = max(1, os.cpu_count() - 1)

from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    Instance,
    Scenario,
    RacingTuner,
    TuningTask,
    build_nsgaii_config_space,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.param_space import PARAM_ROLES
from vamos.engine.tuning.racing.tuning_task import EvalContext
from vamos.experiment.cli.tune import make_evaluator

OUTPUT_CSV = Path(__file__).parent / "fanova_validation.csv"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALGORITHM = "nsgaii"
N_VAR = 30
N_OBJ = 2
REF_POINT = "1.1,1.1"
EVAL_BUDGET = 20_000  # FEs per config run
EVAL_SEEDS = [0, 1]  # seeds for evaluation (averaged)


def _build_eval_fn(problem: str) -> EvalFn:
    return make_evaluator(
        problem_key=problem,
        n_var=N_VAR,
        n_obj=N_OBJ,
        algorithm_name=ALGORITHM,
        fixed_pop_size=100,
        ref_point_str=REF_POINT,
        warm_start=False,
        runtime_penalty=0.0,
        failure_score=0.0,
    )


def _sample_and_evaluate(
    config_space: AlgorithmConfigSpace,
    problem: str,
    n_configs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Sample n_configs random configs, evaluate each, return (configs, scores)."""
    param_space = config_space.to_param_space()
    rng = np.random.default_rng(seed)
    eval_fn = _build_eval_fn(problem)

    configs: list[dict[str, Any]] = []
    scores: list[float] = []

    for i in range(n_configs):
        cfg = param_space.sample(rng)
        # Average over seeds
        seed_scores = []
        for s in EVAL_SEEDS:
            ctx = EvalContext(
                instance=Instance(name=problem, n_var=N_VAR, kwargs={}),
                seed=s,
                budget=EVAL_BUDGET,
            )
            try:
                score = eval_fn(cfg, ctx)
                seed_scores.append(float(score))
            except Exception as e:
                logging.warning(f"Config {i} seed {s} failed: {e}")
                seed_scores.append(0.0)
        avg = float(np.mean(seed_scores))
        configs.append(cfg)
        scores.append(avg)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Evaluated {i + 1}/{n_configs} configs (latest HV={avg:.4f})")

    return configs, scores


def _compute_forest_importance(
    configs: list[dict[str, Any]],
    scores: list[float],
    param_space: Any,
) -> dict[str, float]:
    """Compute per-parameter importance via random forest permutation importance."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("WARNING: scikit-learn not installed. Using variance-based proxy.")
        return _variance_proxy(configs, scores, param_space)

    # Encode configs as feature matrix
    param_names = sorted(param_space.params.keys())
    X = np.zeros((len(configs), len(param_names)))
    for i, cfg in enumerate(configs):
        for j, name in enumerate(param_names):
            val = cfg.get(name, np.nan)
            if isinstance(val, str):
                # Encode categoricals as hash
                X[i, j] = hash(val) % 10000
            elif isinstance(val, bool):
                X[i, j] = float(val)
            else:
                try:
                    X[i, j] = float(val)
                except (TypeError, ValueError):
                    X[i, j] = 0.0
    y = np.array(scores)

    # Filter out NaN rows
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    if len(y) < 10:
        print("WARNING: too few valid samples for forest importance.")
        return {name: 0.0 for name in param_names}

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=N_JOBS)
    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=N_JOBS)

    importances = {}
    for j, name in enumerate(param_names):
        importances[name] = float(result.importances_mean[j])
    return importances


def _variance_proxy(
    configs: list[dict[str, Any]],
    scores: list[float],
    param_space: Any,
) -> dict[str, float]:
    """Fallback: use variance of scores within parameter bins."""
    param_names = sorted(param_space.params.keys())
    importances = {}
    y = np.array(scores)
    total_var = float(np.var(y)) if len(y) > 1 else 1.0

    for name in param_names:
        vals = [cfg.get(name) for cfg in configs]
        # Group by unique values
        unique_vals = list(set(str(v) for v in vals))
        if len(unique_vals) <= 1:
            importances[name] = 0.0
            continue
        groups = {}
        for i, v in enumerate(vals):
            key = str(v)
            groups.setdefault(key, []).append(scores[i])
        # Between-group variance
        group_means = [np.mean(g) for g in groups.values()]
        between_var = float(np.var(group_means)) if len(group_means) > 1 else 0.0
        importances[name] = between_var / max(total_var, 1e-12)

    return importances


def _group_importance_by_role(
    importances: dict[str, float],
    config_space: AlgorithmConfigSpace,
) -> dict[str, float]:
    """Aggregate per-parameter importance by role."""
    role_groups = config_space.group_by_role()
    role_importance: dict[str, float] = {}
    for role in PARAM_ROLES:
        params = role_groups.get(role, [])
        param_names = {p.name for p in params}
        total = sum(importances.get(n, 0.0) for n in param_names)
        role_importance[role] = total
    return role_importance


def run_fanova(n_configs: int = 500, problem: str = "dtlz2", seed: int = 42) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    config_space = build_nsgaii_config_space()
    param_space = config_space.to_param_space()

    print(f"fANOVA Validation: {ALGORITHM} on {problem}")
    print(f"  n_configs={n_configs}, n_var={N_VAR}, n_obj={N_OBJ}")
    print(f"  eval_budget={EVAL_BUDGET}, seeds={EVAL_SEEDS}")
    print("-" * 70)

    t0 = time.perf_counter()
    configs, scores = _sample_and_evaluate(config_space, problem, n_configs, seed)
    wall = time.perf_counter() - t0
    print(f"\nSampling + evaluation took {wall:.1f}s")
    print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}], mean={np.mean(scores):.4f}")

    # Compute importance
    print("\nComputing parameter importance...")
    importances = _compute_forest_importance(configs, scores, param_space)

    # Group by role
    role_importance = _group_importance_by_role(importances, config_space)

    # Print per-parameter results
    print("\n=== Per-Parameter Importance ===")
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    role_groups = config_space.group_by_role()
    param_to_role = {}
    for role, params in role_groups.items():
        for p in params:
            param_to_role[p.name] = role

    for name, imp in sorted_params:
        role = param_to_role.get(name, "unknown")
        print(f"  {name:30s}  {role:15s}  importance={imp:.4f}")

    # Print per-role results
    print("\n=== Per-Role Importance (sum) ===")
    for role in sorted(role_importance, key=lambda r: role_importance[r], reverse=True):
        print(f"  {role:15s}  {role_importance[role]:.4f}")

    # Save CSV
    rows = []
    for name, imp in sorted_params:
        rows.append({
            "parameter": name,
            "role": param_to_role.get(name, "unknown"),
            "importance": f"{imp:.6f}",
            "problem": problem,
            "n_configs": n_configs,
        })
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "role", "importance", "problem", "n_configs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUTPUT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="fANOVA validation for parameter roles")
    parser.add_argument("--n_configs", type=int, default=500, help="Number of random configs to sample")
    parser.add_argument("--problem", type=str, default="zdt1", help="Problem name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with 3 configs")
    args = parser.parse_args()

    n_configs = 3 if args.smoke else args.n_configs
    run_fanova(n_configs=n_configs, problem=args.problem, seed=args.seed)


if __name__ == "__main__":
    main()

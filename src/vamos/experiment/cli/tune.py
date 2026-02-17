from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    EvalContext,
    Instance,
    ModelBasedTuner,
    ParamSpace,
    RacingTuner,
    RandomSearchTuner,
    Scenario,
    TuningTask,
    TrialResult,
    available_model_based_backends,
    build_agemoea_config_space,
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_rvea_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smpso_config_space,
    build_spea2_config_space,
    config_from_assignment,
    filter_active_config,
    save_history_csv,
    save_history_json,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.stats import select_configs_by_paired_test
from vamos.engine.tuning.racing.warm_start import WarmStartEvaluator
from vamos.experiment.unified import optimize
from vamos.foundation.metrics.hypervolume import compute_hypervolume
from vamos.foundation.problem.registry import make_problem_selection

BUILDERS: dict[str, Callable[[], AlgorithmConfigSpace | ParamSpace]] = {
    "nsgaii": build_nsgaii_config_space,
    "nsgaii_permutation": build_nsgaii_permutation_config_space,
    "nsgaii_mixed": build_nsgaii_mixed_config_space,
    "nsgaii_binary": build_nsgaii_binary_config_space,
    "nsgaii_integer": build_nsgaii_integer_config_space,
    "moead": build_moead_config_space,
    "moead_permutation": build_moead_permutation_config_space,
    "moead_binary": build_moead_binary_config_space,
    "moead_integer": build_moead_integer_config_space,
    "nsgaiii": build_nsgaiii_config_space,
    "nsgaiii_binary": build_nsgaiii_binary_config_space,
    "nsgaiii_integer": build_nsgaiii_integer_config_space,
    "spea2": build_spea2_config_space,
    "ibea": build_ibea_config_space,
    "ibea_binary": build_ibea_binary_config_space,
    "ibea_integer": build_ibea_integer_config_space,
    "smpso": build_smpso_config_space,
    "smsemoa": build_smsemoa_config_space,
    "smsemoa_binary": build_smsemoa_binary_config_space,
    "smsemoa_integer": build_smsemoa_integer_config_space,
    "agemoea": build_agemoea_config_space,
    "rvea": build_rvea_config_space,
}

MODEL_BACKENDS = ("optuna", "bohb_optuna", "smac3", "bohb")
NON_MODEL_BACKENDS = ("racing", "random")
ALL_BACKENDS = NON_MODEL_BACKENDS + MODEL_BACKENDS


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _configure_cli_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def _canonical_algorithm_name(name: str) -> str:
    if name in {"nsgaii_permutation", "nsgaii_mixed", "nsgaii_binary", "nsgaii_integer"}:
        return "nsgaii"
    if name in {"moead_permutation", "moead_binary", "moead_integer"}:
        return "moead"
    if name in {"nsgaiii_binary", "nsgaiii_integer"}:
        return "nsgaiii"
    if name in {"smsemoa_binary", "smsemoa_integer"}:
        return "smsemoa"
    if name in {"ibea_binary", "ibea_integer"}:
        return "ibea"
    return name


def _supports_warm_start(name: str) -> bool:
    return _canonical_algorithm_name(name) in {"nsgaii", "moead"}


def _parse_csv_ints(raw: str | None, parser: argparse.ArgumentParser, flag: str, *, min_len: int = 1) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if len(parts) < min_len:
        parser.error(f"{flag} must provide at least {min_len} comma-separated integers.")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        parser.error(f"{flag} must be comma-separated integers.")
    if any(v <= 0 for v in values):
        parser.error(f"{flag} values must be > 0.")
    return values


def _parse_csv_strings(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())


def _parse_seed_spec(raw: str | None, *, default_start: int, default_count: int) -> list[int]:
    if raw is None or not raw.strip():
        return [default_start + i for i in range(default_count)]
    out: list[int] = []
    for token in _parse_csv_strings(raw):
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid seed range token: {token!r}")
            lo = int(parts[0].strip())
            hi = int(parts[1].strip())
            if hi <= lo:
                raise ValueError(f"Invalid seed range {token!r}: end must be > start.")
            out.extend(list(range(lo, hi)))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("Seed specification resolved to empty list.")
    return out


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, int(os.cpu_count() or 1) - 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1 or -1.")
    return int(n_jobs)


def _parse_ref_point(raw: str | None, n_obj: int) -> list[float]:
    if raw:
        try:
            parsed = [float(x.strip()) for x in raw.split(",")]
            if len(parsed) == n_obj:
                return parsed
            _logger().warning("Reference point length (%s) does not match n_obj=%s. Falling back to default.", len(parsed), n_obj)
        except ValueError:
            _logger().warning("Failed to parse --ref-point. Falling back to default.")
    return [1.1] * n_obj


def _build_aggregator(mode: str) -> Callable[[list[float]], float]:
    m = str(mode).strip().lower()
    if m == "mean":
        return lambda scores: float(np.mean(scores))
    if m == "median":
        return lambda scores: float(np.median(scores))
    if m == "p25":
        return lambda scores: float(np.percentile(scores, 25))
    if m == "p10":
        return lambda scores: float(np.percentile(scores, 10))
    raise ValueError(f"Unsupported aggregate mode: {mode!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAMOS tuning CLI (unified backends)")
    parser.add_argument("--problem", type=str, default="zdt1", help="Problem ID (e.g., zdt1).")
    parser.add_argument("--instances", type=str, default="", help="Optional comma-separated problem IDs. Overrides --problem.")
    parser.add_argument("--algorithm", type=str, default="nsgaii", choices=sorted(BUILDERS), help="Algorithm family to tune.")
    parser.add_argument("--backend", type=str, default="optuna", choices=ALL_BACKENDS, help="Tuning backend.")
    parser.add_argument(
        "--backend-fallback",
        type=str,
        default="error",
        choices=["error", "racing", "random"],
        help="Fallback backend if requested model backend is unavailable.",
    )
    parser.add_argument("--list-backends", action="store_true", help="Print backend availability and exit.")

    parser.add_argument("--n-var", type=int, default=30, help="Number of variables.")
    parser.add_argument("--n-obj", type=int, default=2, help="Number of objectives.")
    parser.add_argument("--budget", type=int, default=5000, help="Max evaluations per algorithm run.")
    parser.add_argument("--tune-budget", type=int, default=200, help="Racing experiments or model trials.")
    parser.add_argument("--seed", type=int, default=1, help="Global seed.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds per configuration.")
    parser.add_argument("--validation-seeds", type=str, default="", help="Validation seed list or ranges (e.g., 1001:1011).")
    parser.add_argument("--test-seeds", type=str, default="", help="Test seed list or ranges (e.g., 2001:2011).")
    parser.add_argument("--pop-size", type=int, default=100, help="Fallback fixed population size.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 means CPU cores - 1).")
    parser.add_argument("--ref-point", type=str, default=None, help="HV reference point: comma list, e.g. 1.1,1.1.")
    parser.add_argument(
        "--aggregate-mode",
        type=str,
        default="mean",
        choices=["mean", "median", "p25", "p10"],
        help="Aggregation across instance/seed scores.",
    )
    parser.add_argument(
        "--runtime-penalty",
        type=float,
        default=0.0,
        help="Lambda in score = HV - lambda*log1p(runtime_seconds).",
    )
    parser.add_argument(
        "--failure-score",
        type=float,
        default=0.0,
        help="Score used when an evaluation fails.",
    )

    parser.add_argument("--multi-fidelity", action=argparse.BooleanOptionalAction, default=True, help="Enable multi-fidelity schedule.")
    parser.add_argument("--fidelity-levels", type=str, default=None, help="Comma budgets, e.g. 500,1000,1500.")
    parser.add_argument("--fidelity-promotion-ratio", type=float, default=0.3, help="Promotion ratio for racing multi-fidelity.")
    parser.add_argument("--fidelity-min-configs", type=int, default=3, help="Minimum promoted configs per fidelity level.")
    parser.add_argument("--fidelity-warm-start", dest="fidelity_warm_start", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--initial-configs", type=int, default=20, help="Initial sampled configs for racing.")
    parser.add_argument("--elimination-fraction", type=float, default=0.25, help="Racing elimination fraction.")
    parser.add_argument("--min-blocks-before-elimination", type=int, default=3, help="Racing grace blocks before pruning.")
    parser.add_argument("--use-statistical-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance for racing statistical elimination.")

    parser.add_argument("--bohb-reduction-factor", type=int, default=3, help="Reduction factor for BOHB/Hyperband-style backends.")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Optional wallclock timeout. 0 disables.")
    parser.add_argument("--show-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fidelity-min-instance-frac",
        type=float,
        default=1.0,
        help="Minimum instance fraction at lowest budget for model backends (1.0 disables instance subsampling).",
    )
    parser.add_argument(
        "--fidelity-min-seed-count",
        type=int,
        default=0,
        help="Minimum seed count at lowest budget for model backends (0 uses all seeds).",
    )
    parser.add_argument(
        "--fidelity-max-seed-count",
        type=int,
        default=0,
        help="Maximum seed count at highest budget for model backends (0 uses all seeds).",
    )
    parser.add_argument(
        "--fidelity-selection-seed",
        type=int,
        default=-1,
        help="Seed for deterministic fidelity subsampling (-1 uses --seed).",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default="",
        help="Optional Optuna storage URL for persistent/restartable studies (e.g. sqlite:///results/tune.db).",
    )
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default="",
        help="Optional Optuna study name. Used with --optuna-storage.",
    )
    parser.add_argument(
        "--optuna-load-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --optuna-storage, resume existing study if present.",
    )

    parser.add_argument("--split-seed", type=int, default=42, help="Seed used to split instances into train/validation/test.")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="suite_stratified",
        choices=["suite_stratified", "random"],
        help="Instance split strategy.",
    )
    parser.add_argument("--train-frac", type=float, default=0.6, help="Train fraction for instance split.")
    parser.add_argument("--validation-frac", type=float, default=0.2, help="Validation fraction for instance split.")
    parser.add_argument("--run-validation", action=argparse.BooleanOptionalAction, default=True, help="Evaluate top-k on validation split.")
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=False, help="Evaluate selected configs on test split.")
    parser.add_argument("--validation-budget", type=int, default=0, help="Validation evaluation budget (0 uses --budget).")
    parser.add_argument("--test-budget", type=int, default=0, help="Test evaluation budget (0 uses --budget).")
    parser.add_argument("--validation-topk", type=int, default=5, help="Top-k configs from tuning history to validate.")
    parser.add_argument(
        "--run-statistical-finisher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run statistical finisher on train split top-k candidates before validation/test.",
    )
    parser.add_argument("--finisher-topk", type=int, default=5, help="Top-k candidates for statistical finisher.")
    parser.add_argument("--finisher-min-blocks", type=int, default=3, help="Minimum blocks required for finisher statistical tests.")
    parser.add_argument("--finisher-budget", type=int, default=0, help="Finisher evaluation budget (0 uses --budget).")
    parser.add_argument("--finisher-alpha", type=float, default=0.05, help="Significance level for finisher statistical tests.")
    parser.add_argument(
        "--finisher-use-friedman",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Friedman pre-check before finisher paired tests.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("results") / "tuning")
    parser.add_argument("--name", type=str, default="", help="Optional run name (otherwise auto-generated).")
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.fidelity_levels = _parse_csv_ints(args.fidelity_levels, parser, "--fidelity-levels", min_len=2)
    if args.budget <= 0:
        parser.error("--budget must be > 0.")
    if args.tune_budget <= 0:
        parser.error("--tune-budget must be > 0.")
    if args.n_var <= 0:
        parser.error("--n-var must be > 0.")
    if args.n_obj <= 0:
        parser.error("--n-obj must be > 0.")
    if args.pop_size <= 0:
        parser.error("--pop-size must be > 0.")
    if args.n_seeds <= 0:
        parser.error("--n-seeds must be > 0.")
    if args.validation_topk <= 0:
        parser.error("--validation-topk must be > 0.")
    if args.finisher_topk <= 0:
        parser.error("--finisher-topk must be > 0.")
    if args.finisher_min_blocks <= 0:
        parser.error("--finisher-min-blocks must be > 0.")
    if args.validation_budget < 0 or args.test_budget < 0 or args.finisher_budget < 0:
        parser.error("--validation-budget, --test-budget and --finisher-budget must be >= 0.")
    if not (0.0 < float(args.finisher_alpha) < 1.0):
        parser.error("--finisher-alpha must be in (0, 1).")
    if float(args.runtime_penalty) < 0.0:
        parser.error("--runtime-penalty must be >= 0.")
    if not (0.0 < float(args.fidelity_min_instance_frac) <= 1.0):
        parser.error("--fidelity-min-instance-frac must be in (0, 1].")
    if int(args.fidelity_min_seed_count) < 0:
        parser.error("--fidelity-min-seed-count must be >= 0.")
    if int(args.fidelity_max_seed_count) < 0:
        parser.error("--fidelity-max-seed-count must be >= 0.")
    if int(args.fidelity_min_seed_count) > 0 and int(args.fidelity_max_seed_count) > 0:
        if int(args.fidelity_min_seed_count) > int(args.fidelity_max_seed_count):
            parser.error("--fidelity-min-seed-count cannot exceed --fidelity-max-seed-count.")
    if not (0.0 < float(args.train_frac) < 1.0):
        parser.error("--train-frac must be in (0, 1).")
    if not (0.0 < float(args.validation_frac) < 1.0):
        parser.error("--validation-frac must be in (0, 1).")
    if float(args.train_frac) + float(args.validation_frac) >= 1.0:
        parser.error("--train-frac + --validation-frac must be < 1.")
    return args


def _resolve_output_dir(base: Path, run_name: str, *, problem: str, algorithm: str, backend: str, seed: int) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = run_name.strip() or f"{problem}_{algorithm}_{backend}_seed{seed}_{ts}"
    out = base.expanduser().resolve() / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_evaluator(
    problem_key: str,
    n_var: int,
    n_obj: int,
    algorithm_name: str,
    fixed_pop_size: int,
    ref_point_str: str | None,
    warm_start: bool,
    runtime_penalty: float,
    failure_score: float,
) -> EvalFn:
    ref_point = _parse_ref_point(ref_point_str, n_obj)

    def _score(result: Any, _ctx: EvalContext) -> float:
        F = getattr(result, "F", None)
        base_hv = float(compute_hypervolume(F, ref_point)) if F is not None and len(F) > 0 else float(failure_score)
        elapsed_s = 0.0
        payload = getattr(result, "data", None)
        if isinstance(payload, dict):
            elapsed_raw = payload.get("_elapsed_s", 0.0)
            try:
                elapsed_s = float(elapsed_raw)
            except Exception:
                elapsed_s = 0.0
        penalized = base_hv - float(runtime_penalty) * float(np.log1p(max(0.0, elapsed_s)))
        return float(penalized)

    def _run_algorithm(
        config_dict: Mapping[str, object],
        ctx: EvalContext,
        checkpoint: object | None,
    ) -> tuple[object, object | None]:
        try:
            start_config: dict[str, Any] = dict(config_dict)
            if algorithm_name == "rvea":
                start_config["n_obj"] = n_obj
            elif "pop_size" not in start_config:
                start_config["pop_size"] = fixed_pop_size

            cfg = config_from_assignment(algorithm_name, start_config)
            algo_name = _canonical_algorithm_name(algorithm_name)
            problem_name = str(getattr(ctx.instance, "name", problem_key))
            problem_kwargs = dict(getattr(ctx.instance, "kwargs", {}) or {})
            problem_kwargs.setdefault("n_var", int(n_var))
            problem_kwargs.setdefault("n_obj", int(n_obj))
            selection = make_problem_selection(problem_name, **problem_kwargs)
            t0 = time.perf_counter()
            result = optimize(
                selection.instantiate(),
                algorithm=algo_name,
                algorithm_config=cfg,
                termination=("max_evaluations", int(ctx.budget)),
                seed=int(ctx.seed),
                engine="numpy",
                checkpoint=checkpoint,
            )
            elapsed_s = float(time.perf_counter() - t0)
            payload = getattr(result, "data", None)
            if isinstance(payload, dict):
                payload["_elapsed_s"] = elapsed_s
            return result, result.data.get("checkpoint")
        except Exception:
            _logger().warning("[tune] evaluation failed; assigning score=0.", exc_info=True)

            class _EmptyResult:
                F = None
                data = {"_elapsed_s": 0.0}

            return _EmptyResult(), None

    if warm_start:
        return WarmStartEvaluator(run_fn=_run_algorithm, score_fn=_score)

    def eval_fn(config_dict: dict[str, Any], ctx: EvalContext) -> float:
        result, _ = _run_algorithm(config_dict, ctx, None)
        return _score(result, ctx)

    return eval_fn


def _build_task(
    args: argparse.Namespace,
    param_space: ParamSpace,
    budget_per_run: int,
    *,
    instances: list[Instance] | None = None,
    seeds: list[int] | None = None,
) -> TuningTask:
    if instances is None:
        problem_names = list(_parse_csv_strings(args.instances)) or [str(args.problem)]
        instances = [Instance(name=name, n_var=int(args.n_var), kwargs={}) for name in problem_names]
    if seeds is None:
        seeds = _parse_seed_spec(None, default_start=int(args.seed), default_count=int(args.n_seeds))
    return TuningTask(
        name=f"tune_{args.problem}_{args.algorithm}_{args.backend}",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        aggregator=_build_aggregator(str(args.aggregate_mode)),
        budget_per_run=int(budget_per_run),
        maximize=True,
    )


def _infer_suite(instance_name: str) -> str:
    lower = str(instance_name).strip().lower()
    if lower.startswith("zdt"):
        return "zdt"
    if lower.startswith("dtlz"):
        return "dtlz"
    if lower.startswith("wfg"):
        return "wfg"
    if lower.startswith("uf"):
        return "uf"
    if lower.startswith("cf"):
        return "cf"
    if lower.startswith("re"):
        return "re"
    if lower.startswith("mw"):
        return "mw"
    return "other"


def _split_counts(n: int, train_frac: float, validation_frac: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = max(1, int(round(n * train_frac)))
    n_valid = max(1, int(round(n * validation_frac)))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_valid)
    while n_train + n_valid + n_test > n:
        if n_train > n_valid and n_train > 1:
            n_train -= 1
        elif n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return int(n_train), int(n_valid), int(n_test)


def _split_instances(
    instances: list[Instance],
    *,
    train_frac: float,
    validation_frac: float,
    split_seed: int,
    strategy: str,
) -> tuple[list[Instance], list[Instance], list[Instance], list[dict[str, Any]]]:
    n = len(instances)
    if n == 0:
        raise ValueError("No instances provided for tuning.")
    rng = np.random.default_rng(int(split_seed))
    perm = np.arange(n, dtype=int)
    rng.shuffle(perm)
    ordered = [instances[i] for i in perm]

    # Graceful fallback for tiny instance sets: reuse instance partitions but keep seeds disjoint.
    if n == 1:
        only = [ordered[0]]
        manifest = [
            {
                "instance": ordered[0].name,
                "suite": _infer_suite(ordered[0].name),
                "split": "train/validation/test",
                "shared_instance": True,
            }
        ]
        _logger().warning("Only one instance provided; train/validation/test share instance but use disjoint seeds.")
        return only, only, only, manifest
    if n == 2:
        train = [ordered[0]]
        validation = [ordered[1]]
        test = [ordered[1]]
        manifest = [
            {"instance": ordered[0].name, "suite": _infer_suite(ordered[0].name), "split": "train", "shared_instance": False},
            {"instance": ordered[1].name, "suite": _infer_suite(ordered[1].name), "split": "validation/test", "shared_instance": True},
        ]
        _logger().warning("Only two instances provided; validation and test share one instance but use disjoint seeds.")
        return train, validation, test, manifest

    if str(strategy) == "suite_stratified":
        suite_groups: dict[str, list[Instance]] = {}
        for inst in ordered:
            suite_groups.setdefault(_infer_suite(inst.name), []).append(inst)
        train = []
        validation = []
        test = []
        for suite_name, group in sorted(suite_groups.items()):
            perm_local = np.arange(len(group), dtype=int)
            rng.shuffle(perm_local)
            g = [group[i] for i in perm_local]
            g_train, g_valid, g_test = _split_counts(len(g), train_frac, validation_frac)
            train.extend(g[:g_train])
            validation.extend(g[g_train : g_train + g_valid])
            test.extend(g[g_train + g_valid : g_train + g_valid + g_test])
            _logger().debug(
                "Suite split %s -> train=%s validation=%s test=%s",
                suite_name,
                g_train,
                g_valid,
                g_test,
            )
        # Ensure global non-empty validation/test when possible.
        if not validation and len(train) > 1:
            validation.append(train.pop())
        if not test and len(train) > 1:
            test.append(train.pop())
    else:
        n_train, n_valid, n_test = _split_counts(n, train_frac, validation_frac)
        train = ordered[:n_train]
        validation = ordered[n_train : n_train + n_valid]
        test = ordered[n_train + n_valid : n_train + n_valid + n_test]

    manifest = (
        [{"instance": inst.name, "suite": _infer_suite(inst.name), "split": "train", "shared_instance": False} for inst in train]
        + [{"instance": inst.name, "suite": _infer_suite(inst.name), "split": "validation", "shared_instance": False} for inst in validation]
        + [{"instance": inst.name, "suite": _infer_suite(inst.name), "split": "test", "shared_instance": False} for inst in test]
    )
    return train, validation, test, manifest


def _resolve_split_seeds(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    train = _parse_seed_spec(None, default_start=int(args.seed), default_count=int(args.n_seeds))
    validation = _parse_seed_spec(
        args.validation_seeds,
        default_start=int(args.seed) + 10_000,
        default_count=int(args.n_seeds),
    )
    test = _parse_seed_spec(
        args.test_seeds,
        default_start=int(args.seed) + 20_000,
        default_count=int(args.n_seeds),
    )
    st = set(train)
    sv = set(validation)
    ss = set(test)
    if st & sv or st & ss or sv & ss:
        raise ValueError("Seed splits must be disjoint across train/validation/test.")
    return train, validation, test


def _rank_history_topk(history: list[TrialResult], param_space: ParamSpace, k: int) -> list[dict[str, Any]]:
    if not history:
        return []
    best_by_cfg: dict[str, dict[str, Any]] = {}
    for tr in history:
        active_cfg = filter_active_config(dict(tr.config), param_space)
        cfg_json = json.dumps(active_cfg, sort_keys=True)
        row = best_by_cfg.get(cfg_json)
        score = float(tr.score)
        if row is None or score > float(row["score"]):
            best_by_cfg[cfg_json] = {"score": score, "config": active_cfg}
    ranked = sorted(best_by_cfg.values(), key=lambda d: float(d["score"]), reverse=True)
    return ranked[: max(1, int(k))]


def _evaluate_config_split(
    *,
    config: dict[str, Any],
    eval_fn: EvalFn,
    instances: list[Instance],
    seeds: list[int],
    budget: int,
    aggregator: Callable[[list[float]], float],
) -> dict[str, Any]:
    scores: list[float] = []
    rows_total = 0
    rows_ok = 0
    for inst in instances:
        for seed in seeds:
            rows_total += 1
            ctx = EvalContext(instance=inst, seed=int(seed), budget=int(budget))
            try:
                result = eval_fn(config, ctx)
                if isinstance(result, tuple):
                    score = float(result[0])
                else:
                    score = float(result)
                rows_ok += 1
            except Exception:
                score = float("nan")
            scores.append(score)
    valid = [s for s in scores if np.isfinite(s)]
    agg = float(aggregator(valid)) if valid else float("nan")
    return {
        "score_agg": agg,
        "score_mean": float(np.mean(valid)) if valid else float("nan"),
        "score_median": float(np.median(valid)) if valid else float("nan"),
        "score_p25": float(np.percentile(valid, 25)) if valid else float("nan"),
        "score_p10": float(np.percentile(valid, 10)) if valid else float("nan"),
        "rows_total": int(rows_total),
        "rows_ok": int(rows_ok),
        "fail_rate": float(1.0 - (rows_ok / rows_total if rows_total else 0.0)),
    }


def _append_summary(
    output_dir: Path,
    *,
    summary_updates: dict[str, Any] | None = None,
    artifact_updates: dict[str, str] | None = None,
) -> None:
    summary_path = output_dir / "tuning_summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_updates:
        payload.update(summary_updates)
    if artifact_updates:
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts.update(artifact_updates)
        payload["artifacts"] = artifacts
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_statistical_finisher(
    *,
    candidates: list[dict[str, Any]],
    eval_fn: EvalFn,
    instances: list[Instance],
    seeds: list[int],
    budget: int,
    aggregator: Callable[[list[float]], float],
    alpha: float,
    min_blocks: int,
    failure_score: float,
    use_friedman: bool,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    blocks = [(inst, int(seed)) for inst in instances for seed in seeds]
    if not blocks:
        return None

    n_cfg = len(candidates)
    n_blocks = len(blocks)
    scores = np.full((n_cfg, n_blocks), float(failure_score), dtype=float)
    block_rows: list[dict[str, Any]] = []

    for cfg_idx, row in enumerate(candidates):
        cfg = dict(row["config"])
        for block_idx, (inst, seed) in enumerate(blocks):
            ctx = EvalContext(instance=inst, seed=int(seed), budget=int(budget))
            try:
                result = eval_fn(cfg, ctx)
                score = float(result[0]) if isinstance(result, tuple) else float(result)
            except Exception:
                score = float(failure_score)
            if not np.isfinite(score):
                score = float(failure_score)
            scores[cfg_idx, block_idx] = score
            block_rows.append(
                {
                    "candidate_idx": int(cfg_idx),
                    "instance": str(inst.name),
                    "seed": int(seed),
                    "block_idx": int(block_idx),
                    "score": float(score),
                }
            )

    agg_scores = np.asarray([float(aggregator(scores[i, :].tolist())) for i in range(n_cfg)], dtype=float)
    winner_idx = int(np.argmax(agg_scores))
    keep_mask = np.ones(n_cfg, dtype=bool)
    method = "aggregate_only"
    friedman_pvalue: float | None = None

    if n_cfg >= 2 and n_blocks >= int(min_blocks):
        should_run_paired = True
        if bool(use_friedman) and n_cfg >= 3 and n_blocks >= 3:
            try:
                from scipy.stats import friedmanchisquare  # type: ignore[import-untyped]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    _, p_val = friedmanchisquare(*[scores[i, :] for i in range(n_cfg)])
                if np.isfinite(p_val):
                    friedman_pvalue = float(p_val)
                    if float(p_val) > float(alpha):
                        should_run_paired = False
                        method = "friedman_no_difference"
            except Exception:
                pass

        if should_run_paired:
            keep_mask = select_configs_by_paired_test(
                scores=scores,
                maximize=True,
                alpha=float(alpha),
                aggregator=aggregator,
            )
            if not bool(keep_mask.any()):
                keep_mask[winner_idx] = True
            alive_idx = np.flatnonzero(keep_mask)
            if alive_idx.size > 0:
                best_local = int(np.argmax(agg_scores[alive_idx]))
                winner_idx = int(alive_idx[best_local])
            method = "paired_holm"

    candidate_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        row_scores = scores[idx, :]
        candidate_rows.append(
            {
                "candidate_idx": int(idx),
                "tune_score": float(row["score"]),
                "score_agg": float(agg_scores[idx]),
                "score_mean": float(np.mean(row_scores)),
                "score_median": float(np.median(row_scores)),
                "score_p25": float(np.percentile(row_scores, 25)),
                "score_p10": float(np.percentile(row_scores, 10)),
                "kept_by_test": bool(keep_mask[idx]),
                "selected": bool(idx == winner_idx),
                "config_json": json.dumps(row["config"], sort_keys=True),
            }
        )

    return {
        "winner_idx": int(winner_idx),
        "winner_config": dict(candidates[winner_idx]["config"]),
        "method": str(method),
        "alpha": float(alpha),
        "num_candidates": int(n_cfg),
        "num_blocks": int(n_blocks),
        "friedman_pvalue": (None if friedman_pvalue is None else float(friedman_pvalue)),
        "candidate_rows": candidate_rows,
        "block_rows": block_rows,
    }


def _run_backend(
    args: argparse.Namespace,
    task: TuningTask,
    eval_fn: EvalFn,
    resolved_jobs: int,
) -> tuple[dict[str, Any], list[TrialResult]]:
    fidelity_levels = args.fidelity_levels
    if args.backend in MODEL_BACKENDS:
        min_seed_count = int(args.fidelity_min_seed_count)
        max_seed_count = int(args.fidelity_max_seed_count)
        tuner = ModelBasedTuner(
            task=task,
            max_trials=int(args.tune_budget),
            backend=str(args.backend),
            seed=int(args.seed),
            n_jobs=int(resolved_jobs),
            timeout_seconds=None if float(args.timeout_seconds) <= 0.0 else float(args.timeout_seconds),
            show_progress_bar=bool(args.show_progress_bar),
            bohb_reduction_factor=max(2, int(args.bohb_reduction_factor)),
            budget_levels=list(fidelity_levels) if fidelity_levels else None,
            fidelity_min_instance_frac=float(args.fidelity_min_instance_frac),
            fidelity_min_seed_count=(None if min_seed_count <= 0 else int(min_seed_count)),
            fidelity_max_seed_count=(None if max_seed_count <= 0 else int(max_seed_count)),
            fidelity_selection_seed=(None if int(args.fidelity_selection_seed) < 0 else int(args.fidelity_selection_seed)),
            optuna_storage_url=(str(args.optuna_storage).strip() or None),
            optuna_study_name=(str(args.optuna_study_name).strip() or None),
            optuna_load_if_exists=bool(args.optuna_load_if_exists),
        )
        return tuner.run(eval_fn, verbose=True)

    if args.backend == "random":
        tuner = RandomSearchTuner(task=task, max_trials=int(args.tune_budget), seed=int(args.seed))
        return tuner.run(eval_fn, verbose=True)

    scenario = Scenario(
        max_experiments=int(args.tune_budget),
        elimination_fraction=float(args.elimination_fraction),
        alpha=float(args.alpha),
        min_blocks_before_elimination=int(args.min_blocks_before_elimination),
        use_statistical_tests=bool(args.use_statistical_tests),
        n_jobs=int(resolved_jobs),
        verbose=True,
        use_multi_fidelity=bool(args.multi_fidelity),
        fidelity_levels=tuple(int(v) for v in fidelity_levels) if fidelity_levels else Scenario.fidelity_levels,
        fidelity_promotion_ratio=float(args.fidelity_promotion_ratio),
        fidelity_min_configs=int(args.fidelity_min_configs),
        fidelity_warm_start=bool(args.fidelity_warm_start),
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=int(args.seed), max_initial_configs=int(args.initial_configs))
    return tuner.run(eval_fn, verbose=True)


def _persist_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    task: TuningTask,
    best_config: dict[str, Any],
    history: list[TrialResult],
    elapsed_seconds: float,
    resolved_jobs: int,
) -> None:
    best_active = filter_active_config(best_config, task.param_space)
    best_score = max((float(h.score) for h in history), default=float("nan"))

    best_raw_path = output_dir / "best_config_raw.json"
    best_active_path = output_dir / "best_config_active.json"
    summary_path = output_dir / "tuning_summary.json"
    history_json_path = output_dir / "tuning_history.json"
    history_csv_path = output_dir / "tuning_history.csv"

    best_raw_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    best_active_path.write_text(json.dumps(best_active, indent=2), encoding="utf-8")
    save_history_json(history, task.param_space, history_json_path, include_raw=True)
    save_history_csv(history, task.param_space, history_csv_path, include_raw=True)

    arg_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "schema_version": "vamos_tuning_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": str(args.backend),
        "problem": str(args.problem),
        "algorithm": str(args.algorithm),
        "n_var": int(args.n_var),
        "n_obj": int(args.n_obj),
        "budget_per_run": int(task.budget_per_run),
        "seed": int(args.seed),
        "n_jobs_resolved": int(resolved_jobs),
        "aggregate_mode": str(args.aggregate_mode),
        "runtime_penalty": float(args.runtime_penalty),
        "failure_score": float(args.failure_score),
        "trials_observed": int(len(history)),
        "best_score": float(best_score),
        "elapsed_seconds": float(elapsed_seconds),
        "available_model_backends": available_model_based_backends(),
        "args": arg_payload,
        "artifacts": {
            "best_config_raw": best_raw_path.name,
            "best_config_active": best_active_path.name,
            "tuning_summary": summary_path.name,
            "history_json": history_json_path.name,
            "history_csv": history_csv_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _logger().info("Saved: %s", best_raw_path)
    _logger().info("Saved: %s", best_active_path)
    _logger().info("Saved: %s", summary_path)
    _logger().info("Saved: %s", history_json_path)
    _logger().info("Saved: %s", history_csv_path)


def _print_backend_table() -> None:
    flags = available_model_based_backends()
    print("Backend availability:")
    print("  racing       : True")
    print("  random       : True")
    for name in MODEL_BACKENDS:
        print(f"  {name:12s}: {bool(flags.get(name, False))}")


def main(argv: Sequence[str] | None = None) -> None:
    _configure_cli_logging()
    args = _parse_args(argv)
    if bool(args.list_backends):
        _print_backend_table()
        return

    requested_backend = str(args.backend)
    effective_backend = requested_backend
    availability = available_model_based_backends()
    if requested_backend in MODEL_BACKENDS and not bool(availability.get(requested_backend, False)):
        fallback = str(args.backend_fallback)
        if fallback == "error":
            raise RuntimeError(
                f"Requested backend '{requested_backend}' is not available. "
                f"Install optional dependencies or use --backend-fallback racing/random."
            )
        effective_backend = fallback
        _logger().warning(
            "Backend '%s' unavailable; falling back to '%s'.",
            requested_backend,
            effective_backend,
        )
        args.backend = effective_backend

    resolved_jobs = _resolve_n_jobs(int(args.n_jobs))
    builder = BUILDERS[str(args.algorithm)]
    algo_space = builder()
    param_space = algo_space.to_param_space() if isinstance(algo_space, AlgorithmConfigSpace) else algo_space

    problem_names = list(_parse_csv_strings(args.instances)) or [str(args.problem)]
    all_instances = [Instance(name=name, n_var=int(args.n_var), kwargs={}) for name in problem_names]
    train_instances, validation_instances, test_instances, split_manifest = _split_instances(
        all_instances,
        train_frac=float(args.train_frac),
        validation_frac=float(args.validation_frac),
        split_seed=int(args.split_seed),
        strategy=str(args.split_strategy),
    )
    try:
        train_seeds, validation_seeds, test_seeds = _resolve_split_seeds(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    warm_start_enabled = bool(args.multi_fidelity) and bool(args.fidelity_warm_start)
    if warm_start_enabled and not _supports_warm_start(str(args.algorithm)):
        _logger().warning("Warm-start is not supported for %s; disabling it.", args.algorithm)
        warm_start_enabled = False
    if warm_start_enabled and str(args.backend) in MODEL_BACKENDS:
        _logger().warning("Warm-start is not supported for model-based backends; disabling it.")
        warm_start_enabled = False
    if warm_start_enabled and str(args.backend) == "random":
        _logger().warning("Warm-start is not supported for random backend; disabling it.")
        warm_start_enabled = False

    budget_per_run = int(args.budget)
    if bool(args.multi_fidelity) and args.fidelity_levels:
        budget_per_run = int(max(args.fidelity_levels))
    task = _build_task(
        args,
        param_space,
        budget_per_run=budget_per_run,
        instances=train_instances,
        seeds=train_seeds,
    )
    eval_fn = make_evaluator(
        problem_key=str(args.problem),
        n_var=int(args.n_var),
        n_obj=int(args.n_obj),
        algorithm_name=str(args.algorithm),
        fixed_pop_size=int(args.pop_size),
        ref_point_str=args.ref_point,
        warm_start=warm_start_enabled,
        runtime_penalty=float(args.runtime_penalty),
        failure_score=float(args.failure_score),
    )

    out_dir = _resolve_output_dir(
        args.output_dir,
        str(args.name),
        problem=str(args.problem),
        algorithm=str(args.algorithm),
        backend=str(args.backend),
        seed=int(args.seed),
    )
    _logger().info("Output: %s", out_dir)
    _logger().info("Backend: requested=%s effective=%s", requested_backend, effective_backend)
    _logger().info("Jobs: requested=%s resolved=%s", args.n_jobs, resolved_jobs)
    _logger().info("Tune budget: %s", args.tune_budget)
    _logger().info(
        "Split sizes (instances): train=%s validation=%s test=%s",
        len(train_instances),
        len(validation_instances),
        len(test_instances),
    )
    _logger().info(
        "Split sizes (seeds): train=%s validation=%s test=%s",
        len(train_seeds),
        len(validation_seeds),
        len(test_seeds),
    )

    split_csv_path = out_dir / "split_instances.csv"
    with split_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["instance", "suite", "split", "shared_instance"])
        writer.writeheader()
        writer.writerows(split_manifest)
    split_seed_path = out_dir / "split_seeds.json"
    split_seed_path.write_text(
        json.dumps(
            {
                "train_seeds": train_seeds,
                "validation_seeds": validation_seeds,
                "test_seeds": test_seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    best_config, history = _run_backend(args, task, eval_fn, resolved_jobs=resolved_jobs)
    elapsed = time.perf_counter() - t0

    finisher_summary_updates: dict[str, Any] | None = None
    finisher_artifacts: dict[str, str] | None = None
    if bool(args.run_statistical_finisher):
        finisher_budget = int(args.finisher_budget) if int(args.finisher_budget) > 0 else int(args.budget)
        finisher_candidates = _rank_history_topk(history, task.param_space, int(args.finisher_topk))
        finisher_result = _run_statistical_finisher(
            candidates=finisher_candidates,
            eval_fn=eval_fn,
            instances=train_instances,
            seeds=train_seeds,
            budget=finisher_budget,
            aggregator=task.aggregator,
            alpha=float(args.finisher_alpha),
            min_blocks=int(args.finisher_min_blocks),
            failure_score=float(args.failure_score),
            use_friedman=bool(args.finisher_use_friedman),
        )
        if finisher_result is not None:
            best_config = dict(finisher_result["winner_config"])
            fin_summary_path = out_dir / "statistical_finisher_summary.json"
            fin_candidates_path = out_dir / "statistical_finisher_candidates.csv"
            fin_blocks_path = out_dir / "statistical_finisher_blocks.csv"
            fin_summary_path.write_text(json.dumps(finisher_result, indent=2), encoding="utf-8")
            candidate_rows = list(finisher_result["candidate_rows"])
            block_rows = list(finisher_result["block_rows"])
            with fin_candidates_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(candidate_rows[0].keys()))
                writer.writeheader()
                writer.writerows(candidate_rows)
            with fin_blocks_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(block_rows[0].keys()))
                writer.writeheader()
                writer.writerows(block_rows)
            finisher_summary_updates = {
                "statistical_finisher_ran": True,
                "statistical_finisher_method": str(finisher_result["method"]),
                "statistical_finisher_budget": int(finisher_budget),
            }
            finisher_artifacts = {
                "statistical_finisher_summary": fin_summary_path.name,
                "statistical_finisher_candidates": fin_candidates_path.name,
                "statistical_finisher_blocks": fin_blocks_path.name,
            }
            _logger().info(
                "Statistical finisher selected candidate %s using method '%s'.",
                finisher_result["winner_idx"],
                finisher_result["method"],
            )

    _logger().info("--- Tuning complete ---")
    _logger().info("Best configuration:")
    for k, v in best_config.items():
        _logger().info("  %s: %s", k, v)

    _persist_artifacts(
        output_dir=out_dir,
        args=args,
        task=task,
        best_config=best_config,
        history=history,
        elapsed_seconds=elapsed,
        resolved_jobs=resolved_jobs,
    )
    if finisher_summary_updates or finisher_artifacts:
        _append_summary(
            out_dir,
            summary_updates=finisher_summary_updates,
            artifact_updates=finisher_artifacts,
        )
    _append_summary(
        out_dir,
        summary_updates={
            "backend_requested": requested_backend,
            "backend_effective": effective_backend,
            "split": {
                "instance_counts": {
                    "train": len(train_instances),
                    "validation": len(validation_instances),
                    "test": len(test_instances),
                },
                "seed_counts": {
                    "train": len(train_seeds),
                    "validation": len(validation_seeds),
                    "test": len(test_seeds),
                },
                "split_seed": int(args.split_seed),
                "split_strategy": str(args.split_strategy),
                "train_frac": float(args.train_frac),
                "validation_frac": float(args.validation_frac),
            }
        },
        artifact_updates={
            "split_instances": split_csv_path.name,
            "split_seeds": split_seed_path.name,
        },
    )

    validation_rows: list[dict[str, Any]] = []
    champions: dict[str, dict[str, Any]] = {}
    if bool(args.run_validation):
        _logger().info("Running validation split evaluation...")
        val_budget = int(args.validation_budget) if int(args.validation_budget) > 0 else int(args.budget)
        ranked = _rank_history_topk(history, task.param_space, int(args.validation_topk))
        for rank, row in enumerate(ranked, start=1):
            metrics = _evaluate_config_split(
                config=dict(row["config"]),
                eval_fn=eval_fn,
                instances=validation_instances,
                seeds=validation_seeds,
                budget=val_budget,
                aggregator=task.aggregator,
            )
            validation_rows.append(
                {
                    "rank": int(rank),
                    "tune_score": float(row["score"]),
                    "config_json": json.dumps(row["config"], sort_keys=True),
                    **metrics,
                }
            )
        validation_rows.sort(key=lambda d: float(d["score_agg"]), reverse=True)
        val_csv_path = out_dir / "validation_metrics.csv"
        with val_csv_path.open("w", encoding="utf-8", newline="") as fh:
            if validation_rows:
                writer = csv.DictWriter(fh, fieldnames=list(validation_rows[0].keys()))
                writer.writeheader()
                writer.writerows(validation_rows)
            else:
                writer = csv.writer(fh)
                writer.writerow(["rank", "tune_score", "config_json", "score_agg", "score_mean", "score_median", "score_p25", "score_p10", "rows_total", "rows_ok", "fail_rate"])

        if validation_rows:
            best_global = validation_rows[0]
            best_robust = sorted(validation_rows, key=lambda d: float(d["score_p25"]), reverse=True)[0]
            best_fast = sorted(validation_rows, key=lambda d: float(d["score_mean"]), reverse=True)[0]
            champions = {
                "champion_global": json.loads(str(best_global["config_json"])),
                "champion_robust": json.loads(str(best_robust["config_json"])),
                "champion_fast": json.loads(str(best_fast["config_json"])),
            }
            champions_path = out_dir / "selected_configs_validation.json"
            champions_path.write_text(json.dumps(champions, indent=2), encoding="utf-8")
            _append_summary(
                out_dir,
                summary_updates={"validation_ran": True, "validation_budget": int(val_budget)},
                artifact_updates={
                    "validation_metrics": val_csv_path.name,
                    "selected_configs_validation": champions_path.name,
                },
            )
        else:
            _append_summary(
                out_dir,
                summary_updates={"validation_ran": True, "validation_budget": int(val_budget)},
                artifact_updates={"validation_metrics": val_csv_path.name},
            )

    if bool(args.run_test):
        _logger().info("Running test split evaluation...")
        test_budget = int(args.test_budget) if int(args.test_budget) > 0 else int(args.budget)
        if not champions:
            champions = {"champion_global": filter_active_config(dict(best_config), task.param_space)}
        unique_candidates: dict[str, dict[str, Any]] = {}
        for name, cfg in champions.items():
            unique_candidates[name] = dict(cfg)
        test_rows: list[dict[str, Any]] = []
        for label, cfg in unique_candidates.items():
            metrics = _evaluate_config_split(
                config=dict(cfg),
                eval_fn=eval_fn,
                instances=test_instances,
                seeds=test_seeds,
                budget=test_budget,
                aggregator=task.aggregator,
            )
            test_rows.append(
                {
                    "candidate": str(label),
                    "config_json": json.dumps(cfg, sort_keys=True),
                    **metrics,
                }
            )
        test_rows.sort(key=lambda d: float(d["score_agg"]), reverse=True)
        test_csv_path = out_dir / "test_metrics.csv"
        with test_csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(test_rows[0].keys()))
            writer.writeheader()
            writer.writerows(test_rows)
        _append_summary(
            out_dir,
            summary_updates={"test_ran": True, "test_budget": int(test_budget)},
            artifact_updates={"test_metrics": test_csv_path.name},
        )


if __name__ == "__main__":
    main()

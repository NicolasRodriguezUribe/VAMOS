from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    EvalContext,
    Instance,
    ModelBasedTuner,
    MOEATuner,
    MOEATunerConfig,
    ParamSpace,
    RacingTuner,
    RandomSearchTuner,
    Scenario,
    TrialResult,
    TuningTask,
    available_model_based_backends,
    build_agemoea_binary_config_space,
    build_agemoea_config_space,
    build_agemoea_integer_config_space,
    build_agemoea_permutation_config_space,
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_ibea_permutation_config_space,
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
    build_nsgaiii_permutation_config_space,
    build_rvea_binary_config_space,
    build_rvea_config_space,
    build_rvea_integer_config_space,
    build_rvea_permutation_config_space,
    build_smpso_config_space,
    build_smpso_mixed_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_smsemoa_permutation_config_space,
    build_spea2_binary_config_space,
    build_spea2_config_space,
    build_spea2_integer_config_space,
    build_spea2_permutation_config_space,
    config_from_assignment,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.warm_start import WarmStartEvaluator
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.quality_indicators.hypervolume import compute_hypervolume

from ._tune_args import build_parser as _build_parser_impl
from ._tune_args import parse_args as _parse_args_impl
from ._tune_flow import (
    run_test_stage as _run_test_stage,
)
from ._tune_flow import (
    run_tuning_with_finisher as _run_tuning_with_finisher,
)
from ._tune_flow import (
    run_validation_stage as _run_validation_stage,
)
from ._tune_flow import (
    write_split_artifacts as _write_split_artifacts,
)
from ._tune_post import (
    append_summary as _append_summary,
)
from ._tune_post import (
    persist_artifacts as _persist_artifacts_impl,
)
from ._tune_post import (
    run_statistical_finisher as _run_statistical_finisher_impl,
)
from ._tune_utils import (
    build_aggregator as _build_aggregator,
)
from ._tune_utils import (
    parse_csv_ints as _parse_csv_ints,
)
from ._tune_utils import (
    parse_csv_strings as _parse_csv_strings,
)
from ._tune_utils import (
    parse_ref_point as _parse_ref_point,
)
from ._tune_utils import (
    parse_seed_spec as _parse_seed_spec,
)
from ._tune_utils import (
    resolve_n_jobs as _resolve_n_jobs,
)
from ._tune_utils import (
    resolve_split_seeds as _resolve_split_seeds,
)
from ._tune_utils import (
    split_instances as _split_instances,
)

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
    "nsgaiii_permutation": build_nsgaiii_permutation_config_space,
    "nsgaiii_binary": build_nsgaiii_binary_config_space,
    "nsgaiii_integer": build_nsgaiii_integer_config_space,
    "spea2": build_spea2_config_space,
    "spea2_permutation": build_spea2_permutation_config_space,
    "spea2_binary": build_spea2_binary_config_space,
    "spea2_integer": build_spea2_integer_config_space,
    "ibea": build_ibea_config_space,
    "ibea_permutation": build_ibea_permutation_config_space,
    "ibea_binary": build_ibea_binary_config_space,
    "ibea_integer": build_ibea_integer_config_space,
    "smpso": build_smpso_config_space,
    "smpso_mixed": build_smpso_mixed_config_space,
    "smsemoa": build_smsemoa_config_space,
    "smsemoa_permutation": build_smsemoa_permutation_config_space,
    "smsemoa_binary": build_smsemoa_binary_config_space,
    "smsemoa_integer": build_smsemoa_integer_config_space,
    "agemoea": build_agemoea_config_space,
    "agemoea_permutation": build_agemoea_permutation_config_space,
    "agemoea_binary": build_agemoea_binary_config_space,
    "agemoea_integer": build_agemoea_integer_config_space,
    "rvea": build_rvea_config_space,
    "rvea_permutation": build_rvea_permutation_config_space,
    "rvea_binary": build_rvea_binary_config_space,
    "rvea_integer": build_rvea_integer_config_space,
}

MODEL_BACKENDS = ("optuna", "bohb_optuna", "smac3", "bohb")
NON_MODEL_BACKENDS = ("racing", "random", "moea_tuner")
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
    return canonical_algorithm_name(name)


def _supports_warm_start(name: str) -> bool:
    return _canonical_algorithm_name(name) in {"nsgaii", "moead"}


def _build_parser() -> argparse.ArgumentParser:
    return _build_parser_impl(builders=BUILDERS, all_backends=ALL_BACKENDS)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _parse_args_impl(
        argv,
        builders=BUILDERS,
        all_backends=ALL_BACKENDS,
        parse_csv_ints=_parse_csv_ints,
    )


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
        model_tuner = ModelBasedTuner(
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
        return model_tuner.run(cast(Callable[[dict[str, Any], EvalContext], float], eval_fn), verbose=True)

    if args.backend == "random":
        return RandomSearchTuner(task=task, max_trials=int(args.tune_budget), seed=int(args.seed)).run(
            cast(Callable[[dict[str, Any], EvalContext], float], eval_fn),
            verbose=True,
        )

    if args.backend == "moea_tuner":
        # Resolve the AlgorithmConfigSpace (not just ParamSpace)
        builder = BUILDERS.get(str(args.algorithm))
        config_space_obj = builder() if builder else None
        if not isinstance(config_space_obj, AlgorithmConfigSpace):
            _logger().warning("[tune] moea_tuner requires AlgorithmConfigSpace; falling back to racing.")
        else:
            kb_enabled = bool(getattr(args, "use_knowledge_base", False))
            kb_path = getattr(args, "knowledge_base_path", None)
            moea_config = MOEATunerConfig(
                max_experiments=int(args.tune_budget),
                max_initial_configs=int(args.initial_configs),
                seed=int(args.seed),
                n_jobs=int(resolved_jobs),
                use_knowledge_base=kb_enabled,
                knowledge_base_path=str(kb_path) if kb_path else None,
            )
            moea_tuner = MOEATuner(
                config_space=config_space_obj,
                task=task,
                config=moea_config,
            )
            return moea_tuner.run(eval_fn, verbose=True)

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
    return RacingTuner(task=task, scenario=scenario, seed=int(args.seed), max_initial_configs=int(args.initial_configs)).run(
        eval_fn,
        verbose=True,
    )


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
    return _run_statistical_finisher_impl(
        candidates=candidates,
        eval_fn=eval_fn,
        instances=instances,
        seeds=seeds,
        budget=budget,
        aggregator=aggregator,
        alpha=alpha,
        min_blocks=min_blocks,
        failure_score=failure_score,
        use_friedman=use_friedman,
    )


def _persist_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    task: TuningTask,
    best_config: dict[str, Any],
    history: list[TrialResult],
    elapsed_seconds: float,
    resolved_jobs: int,
) -> None:
    _persist_artifacts_impl(
        output_dir=output_dir,
        args=args,
        task=task,
        best_config=best_config,
        history=history,
        elapsed_seconds=elapsed_seconds,
        resolved_jobs=resolved_jobs,
        available_backends_fn=available_model_based_backends,
    )


def _print_backend_table() -> None:
    flags = available_model_based_backends()
    print("Backend availability:")
    print("  racing       : True")
    print("  random       : True")
    print("  moea_tuner   : True")
    for name in MODEL_BACKENDS:
        print(f"  {name:12s}: {bool(flags.get(name, False))}")


def main(argv: Sequence[str] | None = None) -> None:
    _configure_cli_logging()
    args = _parse_args(argv)
    if bool(args.list_backends):
        _print_backend_table()
        return

    logger = _logger()
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
        logger.warning(
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
    logger.info("Output: %s", out_dir)
    logger.info("Backend: requested=%s effective=%s", requested_backend, effective_backend)
    logger.info("Jobs: requested=%s resolved=%s", args.n_jobs, resolved_jobs)
    logger.info("Tune budget: %s", args.tune_budget)
    logger.info(
        "Split sizes (instances): train=%s validation=%s test=%s",
        len(train_instances),
        len(validation_instances),
        len(test_instances),
    )
    logger.info(
        "Split sizes (seeds): train=%s validation=%s test=%s",
        len(train_seeds),
        len(validation_seeds),
        len(test_seeds),
    )

    split_csv_path, split_seed_path = _write_split_artifacts(
        out_dir,
        split_manifest,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    best_config, history, elapsed, finisher_summary_updates, finisher_artifacts = _run_tuning_with_finisher(
        args=args,
        task=task,
        eval_fn=eval_fn,
        resolved_jobs=resolved_jobs,
        train_instances=train_instances,
        train_seeds=train_seeds,
        out_dir=out_dir,
        run_backend_fn=_run_backend,
        logger=logger,
    )

    logger.info("--- Tuning complete ---")
    logger.info("Best configuration:")
    for k, v in best_config.items():
        logger.info("  %s: %s", k, v)

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
            },
        },
        artifact_updates={
            "split_instances": split_csv_path.name,
            "split_seeds": split_seed_path.name,
        },
    )

    champions = _run_validation_stage(
        args=args,
        out_dir=out_dir,
        history=history,
        task=task,
        eval_fn=eval_fn,
        validation_instances=validation_instances,
        validation_seeds=validation_seeds,
        logger=logger,
    )
    _run_test_stage(
        args=args,
        out_dir=out_dir,
        champions=champions,
        best_config=best_config,
        task=task,
        eval_fn=eval_fn,
        test_instances=test_instances,
        test_seeds=test_seeds,
        logger=logger,
    )


if __name__ == "__main__":
    main()

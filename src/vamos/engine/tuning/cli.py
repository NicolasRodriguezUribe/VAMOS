from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .core.spec import run_experiment_from_file, load_experiment_spec, build_experiment_from_spec
from .core.tuning_task import TuningTask, EvalContext
from .core.experiment import ExperimentResult


def _load_callable(target: str) -> Callable[..., Any]:
    """
    Load a Python callable from a 'module_path:attr_name' string.
    """
    if ":" not in target:
        raise ValueError(f"Expected 'module:attr' format, got {target!r}")

    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)

    if not callable(obj):
        raise ValueError(f"Loaded object {module_name}:{attr_name} is not callable")

    return obj


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the VAMOS tuning CLI.
    """
    parser = argparse.ArgumentParser(
        prog="vamos.engine.tuning",
        description="Run VAMOS tuning experiments from a JSON specification.",
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the JSON experiment specification file.",
    )

    parser.add_argument(
        "--task-factory",
        "-t",
        required=True,
        help=(
            "Fully qualified callable to create the TuningTask, in the form "
            "'module.path:factory_name'. The callable must return a TuningTask instance."
        ),
    )

    parser.add_argument(
        "--eval-fn",
        "-e",
        required=True,
        help=(
            "Fully qualified evaluation function, in the form "
            "'module.path:func_name'. The function must have signature "
            "eval_fn(config: Dict[str, Any], ctx: EvalContext) -> float."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for the experiment (overrides spec.seed).",
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation suite even if specified in the JSON.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Optional path to write a JSON with the best config and basic summary. "
            "If omitted, results are printed only to stdout."
        ),
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce verbosity (suppress most progress messages).",
    )

    return parser


def _print_experiment_summary(result: ExperimentResult) -> None:
    """
    Print a concise summary of the experiment result to stdout.
    """
    print(f"[result] Experiment name: {result.name}")
    print(f"[result] Tuner kind: {result.tuner_kind}")
    print(f"[result] Number of tuning trials: {len(result.tuning_history)}")

    if result.tuning_history:
        try:
            scores = [t.score for t in result.tuning_history if hasattr(t, "score")]
            if scores:
                best_score = max(scores)
                print(f"[result] Best tuning score: {best_score:.6f}")
        except Exception:
            pass

    print("[result] Best configuration (keys):")
    keys_preview = list(result.best_config.keys())
    print(f"         {keys_preview}")

    if result.benchmark_summaries:
        print("[result] Validation summary (rank 1 = best):")
        for s in result.benchmark_summaries:
            print(
                f"  rank={s.rank} label={s.label} "
                f"mean={s.mean_score:.4f} std={s.std_score:.4f} "
                f"min={s.min_score:.4f} max={s.max_score:.4f} runs={s.num_runs}"
            )

    if result.benchmark_stats:
        print(f"[result] Best config by mean score: {result.benchmark_stats.best_label}")
        print(f"[result] Statistically worse (alpha=0.05): {result.benchmark_stats.worse_labels}")
        print(f"[result] Not significantly worse: {result.benchmark_stats.non_worse_labels}")


def _build_output_payload(result: ExperimentResult) -> Dict[str, Any]:
    """
    Build a JSON-serializable dictionary summarizing the experiment result.
    """
    payload: Dict[str, Any] = {
        "name": result.name,
        "tuner_kind": str(result.tuner_kind),
        "best_config": result.best_config,
    }

    history: list[Dict[str, Any]] = []
    for t in result.tuning_history:
        entry: Dict[str, Any] = {}
        if hasattr(t, "trial_id"):
            entry["trial_id"] = getattr(t, "trial_id")
        if hasattr(t, "score"):
            entry["score"] = getattr(t, "score")
        history.append(entry)
    payload["tuning_history"] = history

    if result.benchmark_summaries is not None:
        payload["benchmark_summary"] = [
            {
                "label": s.label,
                "rank": s.rank,
                "mean_score": s.mean_score,
                "std_score": s.std_score,
                "min_score": s.min_score,
                "max_score": s.max_score,
                "num_runs": s.num_runs,
            }
            for s in result.benchmark_summaries
        ]

    if result.benchmark_stats is not None:
        payload["benchmark_stats"] = {
            "best_label": result.benchmark_stats.best_label,
            "worse_labels": result.benchmark_stats.worse_labels,
            "non_worse_labels": result.benchmark_stats.non_worse_labels,
        }

    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the VAMOS tuning CLI.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)

    try:
        spec = load_experiment_spec(config_path)
    except Exception as exc:
        print(f"[error] Failed to load experiment spec: {exc!r}", file=sys.stderr)
        return 1

    if args.seed is not None:
        spec.seed = int(args.seed)
    if args.no_validation and spec.validation is not None:
        spec.validation.enabled = False

    try:
        task_factory = _load_callable(args.task_factory)
    except Exception as exc:
        print(f"[error] Failed to load task factory {args.task_factory!r}: {exc!r}", file=sys.stderr)
        return 1

    try:
        eval_fn = _load_callable(args.eval_fn)
    except Exception as exc:
        print(f"[error] Failed to load eval_fn {args.eval_fn!r}: {exc!r}", file=sys.stderr)
        return 1

    try:
        task = task_factory()
        if not isinstance(task, TuningTask):
            print(
                f"[error] Task factory {args.task_factory!r} did not return a TuningTask (got {type(task)} instead)",
                file=sys.stderr,
            )
            return 1
    except Exception as exc:
        print(f"[error] Task factory raised an exception: {exc!r}", file=sys.stderr)
        return 1

    try:
        experiment = build_experiment_from_spec(spec, task)
    except Exception as exc:
        print(f"[error] Failed to build experiment from spec: {exc!r}", file=sys.stderr)
        return 1

    verbose = not args.quiet

    if verbose:
        print(f"[cli] Running experiment '{experiment.name}' (tuner_kind={experiment.tuner_kind})")

    try:
        result = experiment.run(eval_fn, verbose=verbose)
    except Exception as exc:
        print(f"[error] Experiment run failed: {exc!r}", file=sys.stderr)
        return 1

    if verbose:
        _print_experiment_summary(result)

    if args.output:
        output_path = Path(args.output)
        try:
            payload = _build_output_payload(result)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            if verbose:
                print(f"[cli] Wrote result summary to {output_path}")
        except Exception as exc:
            print(f"[error] Failed to write output JSON: {exc!r}", file=sys.stderr)
            return 1

    return 0


def example_cli_usage() -> None:
    """
    Example usage of the VAMOS tuning CLI.
    """
    # python -m vamos.engine.tuning \
    #   --config experiment_zdt1.json \
    #   --task-factory myproj.problems.zdt:create_task \
    #   --eval-fn myproj.problems.zdt:eval_fn \
    #   --seed 42 \
    #   --output result_zdt1.json


__all__ = ["main"]

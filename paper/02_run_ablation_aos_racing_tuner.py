"""
VAMOS Paper Ablation: AOS + Racing Tuner
=======================================
Runs a focused ablation within VAMOS to support the paper claims about:
  - Adaptive Operator Selection (AOS)
  - irace-inspired racing tuner

This script uses the same benchmark suite as the runtime experiments (no ZDT5).

Usage:
  python paper/02_run_ablation_aos_racing_tuner.py

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 100000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_N_JOBS: joblib workers (default: CPU count - 1)
  - VAMOS_ABLATION_ENGINE: VAMOS engine for evaluation runs (default: numba)
  - VAMOS_ABLATION_VARIANTS: comma-separated variants to run
      (default: baseline,aos,tuned,tuned_aos)
  - VAMOS_ABLATION_OUTPUT_CSV: output CSV path
      (default: experiments/ablation_aos_racing_tuner.csv)
  - VAMOS_ABLATION_ANYTIME_CSV: optional checkpoint CSV output path
      (default: experiments/ablation_aos_anytime.csv; set empty/0 to disable)
  - VAMOS_ABLATION_CHECKPOINTS: comma-separated evaluation checkpoints for anytime HV
      (default: 5000,10000,20000,50000,100000)

Tuner controls (optional):
  - VAMOS_TUNER_ENABLE: 1/0 (default: 1 if "tuned" variant requested)
  - VAMOS_TUNER_PROBLEMS: comma-separated training problems
      (default: zdt4,zdt6,dtlz2,dtlz6,dtlz7,wfg1,wfg9)
  - VAMOS_TUNER_N_EVALS: evaluations per tuning run (default: 20000; used only if multi-fidelity is disabled)
  - VAMOS_TUNER_N_SEEDS: number of seeds per tuning block (default: 10)
  - VAMOS_TUNER_SEED0: first seed for tuning (default: 1)
  - VAMOS_TUNER_MAX_EXPERIMENTS: max config×instance×seed blocks (default: 8000)
  - VAMOS_TUNER_N_JOBS: parallel jobs for tuner evaluations (default: 1)
  - VAMOS_TUNER_MAX_INITIAL_CONFIGS: number of configurations sampled per tuning run (default: 60)
  - VAMOS_TUNER_REPEATS: repeat tuning runs with different tuner seeds (default: 5)
  - VAMOS_TUNER_PICK: which repeated run to select ("best" or "median"; default: best)
  - VAMOS_TUNER_MIN_POP_SIZE: minimum pop_size considered during tuning (default: 100)
  - VAMOS_TUNER_USE_MULTI_FIDELITY: 1/0 enable multi-fidelity tuning (default: 1)
  - VAMOS_TUNER_FIDELITY_LEVELS: comma-separated budgets for multi-fidelity tuning
      (default: 20000,60000,100000)
  - VAMOS_TUNER_FIDELITY_WARM_START: 1/0 warm-start between fidelity levels (default: 0)
  - VAMOS_TUNER_OUTPUT_JSON: tuned config path (default: experiments/tuned_nsgaii.json)
  - VAMOS_TUNER_OUTPUT_RESOLVED_JSON: resolved NSGA-II config path
      (default: experiments/tuned_nsgaii_resolved.json)
  - VAMOS_TUNER_HISTORY_CSV: tuner history CSV (default: experiments/tuner_history.csv)
  - VAMOS_TUNER_RUNS_CSV: tuning repeats summary CSV (default: experiments/tuned_nsgaii_runs.csv)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning import (
    EvalContext,
    Instance,
    RacingTuner,
    Scenario,
    TuningTask,
    build_nsgaii_config_space,
    config_from_assignment,
    save_history_csv,
    save_history_json,
)
from vamos.foundation.problem.registry import make_problem_selection

try:
    from .benchmark_utils import compute_hv
except ImportError:  # pragma: no cover
    from benchmark_utils import compute_hv

try:
    from .progress_utils import ProgressBar, joblib_progress
except ImportError:  # pragma: no cover
    from progress_utils import ProgressBar, joblib_progress


# =============================================================================
# CONFIGURATION (match the paper benchmark suite)
# =============================================================================

ZDT_PROBLEMS = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
DTLZ_PROBLEMS = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
WFG_PROBLEMS = ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
WFG_N_VAR = 24

ZDT_N_OBJ = 2
DTLZ_N_OBJ = 3
WFG_N_OBJ = 2

POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0


def problem_dims(problem_name: str) -> tuple[int, int]:
    if problem_name in ZDT_N_VAR:
        return ZDT_N_VAR[problem_name], ZDT_N_OBJ
    if problem_name in DTLZ_N_VAR:
        return DTLZ_N_VAR[problem_name], DTLZ_N_OBJ
    if problem_name in WFG_PROBLEMS:
        return WFG_N_VAR, WFG_N_OBJ
    raise ValueError(f"Unknown problem dimensions for '{problem_name}'")


def _as_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _as_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None else str(raw)


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _maybe_dataclass_to_dict(value: object) -> object:
    return asdict(value) if is_dataclass(value) else value


# =============================================================================
# AOS setup
# =============================================================================


class HVCheckpointRecorder:
    def __init__(self, *, problem_name: str, checkpoints: list[int], start_time: float):
        self.problem_name = str(problem_name).lower()
        self.checkpoints = sorted(set(int(c) for c in checkpoints))
        self._start_time = float(start_time)
        self._next_idx = 0
        self._records: list[dict[str, float]] = []
        self._last_front = None

    def on_start(self, ctx: object | None = None) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F=None,
        X=None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if self._next_idx >= len(self.checkpoints):
            return
        if stats is None:
            return
        evals = stats.get("evals")
        if evals is None:
            return
        try:
            evals_int = int(evals)
        except Exception:
            return
        if F is not None:
            self._last_front = F
        while self._next_idx < len(self.checkpoints) and evals_int >= self.checkpoints[self._next_idx]:
            front = F if F is not None else self._last_front
            hv = compute_hv(front, self.problem_name) if front is not None else 0.0
            seconds = time.perf_counter() - self._start_time
            self._records.append({"evals": float(self.checkpoints[self._next_idx]), "seconds": float(seconds), "hypervolume": float(hv)})
            self._next_idx += 1

    def on_end(self, final_F=None, final_stats: dict[str, Any] | None = None) -> None:
        if self._next_idx >= len(self.checkpoints):
            return
        front = final_F if final_F is not None else self._last_front
        if front is None:
            return
        seconds = time.perf_counter() - self._start_time
        hv = compute_hv(front, self.problem_name)
        while self._next_idx < len(self.checkpoints):
            self._records.append({"evals": float(self.checkpoints[self._next_idx]), "seconds": float(seconds), "hypervolume": float(hv)})
            self._next_idx += 1

    def records(self) -> list[dict[str, float]]:
        return list(self._records)


def make_aos_cfg(*, seed: int, n_var: int) -> dict[str, Any]:
    """
    Minimal, reproducible AOS configuration:
    - enabled
    - deterministic RNG seeded per run
    - small operator portfolio (>=2 arms, otherwise AOS is a no-op)
    """
    return {
        "enabled": True,
        # Epsilon-greedy provides conservative exploration and avoids forcing
        # every arm early, which can be harmful when some operators are highly
        # problem-dependent.
        "method": str(os.environ.get("VAMOS_AOS_METHOD", "epsilon_greedy")),
        "epsilon": float(os.environ.get("VAMOS_AOS_EPSILON", "0.05")),
        "c": float(os.environ.get("VAMOS_AOS_UCB_C", "1.0")),
        "window_size": int(os.environ.get("VAMOS_AOS_WINDOW_SIZE", "50")),
        "min_usage": int(os.environ.get("VAMOS_AOS_MIN_USAGE", "0")),
        "rng_seed": int(seed),
        "reward_scope": "combined",
        "reward_weights": {
            "survival": float(os.environ.get("VAMOS_AOS_W_SURVIVAL", "0.40")),
            "nd_insertions": float(os.environ.get("VAMOS_AOS_W_ND", "0.40")),
            "hv_delta": float(os.environ.get("VAMOS_AOS_W_HV_DELTA", "0.20")),
        },
        "operator_pool": [
            {
                "crossover": ("sbx", {"prob": CROSSOVER_PROB, "eta": CROSSOVER_ETA}),
                "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA}),
            },
            {
                "crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.2}),
                "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA}),
            },
        ],
    }


# =============================================================================
# Tuning setup
# =============================================================================


def tune_nsgaii(*, train_problems: list[str], seed0: int) -> dict[str, Any]:
    """
    Run the racing tuner and return the selected configuration.

    By default we repeat the tuning run multiple times (with different tuner RNG
    seeds) to reduce variance from sampling and racing randomness.
    """
    tune_n_evals = _as_int_env("VAMOS_TUNER_N_EVALS", 20000)
    tune_n_seeds = _as_int_env("VAMOS_TUNER_N_SEEDS", 10)
    tune_max_experiments = _as_int_env("VAMOS_TUNER_MAX_EXPERIMENTS", 8000)
    tune_n_jobs = _as_int_env("VAMOS_TUNER_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
    tune_max_initial_configs = _as_int_env("VAMOS_TUNER_MAX_INITIAL_CONFIGS", 60)
    tune_repeats = _as_int_env("VAMOS_TUNER_REPEATS", 5)
    tune_pick = _as_str_env("VAMOS_TUNER_PICK", "best").strip().lower()
    min_pop_size = _as_int_env("VAMOS_TUNER_MIN_POP_SIZE", 100)
    use_multi_fidelity = _as_int_env("VAMOS_TUNER_USE_MULTI_FIDELITY", 1) != 0
    warm_start = _as_int_env("VAMOS_TUNER_FIDELITY_WARM_START", 0) != 0

    fidelity_levels: list[int] = []
    if use_multi_fidelity:
        raw_levels = _as_str_env("VAMOS_TUNER_FIDELITY_LEVELS", "20000,60000,100000")
        fidelity_levels = _parse_int_list(raw_levels)
        if not fidelity_levels:
            raise ValueError("VAMOS_TUNER_FIDELITY_LEVELS must contain at least one integer budget.")
        if any(b <= 0 for b in fidelity_levels):
            raise ValueError(f"Invalid fidelity budgets (must be positive): {fidelity_levels}")
        fidelity_levels = sorted(set(fidelity_levels))
        if len(fidelity_levels) < 2:
            raise ValueError("VAMOS_TUNER_FIDELITY_LEVELS must contain at least two increasing budgets.")

    if tune_max_initial_configs < 1:
        raise ValueError("VAMOS_TUNER_MAX_INITIAL_CONFIGS must be >= 1.")
    if tune_repeats < 1:
        raise ValueError("VAMOS_TUNER_REPEATS must be >= 1.")
    if tune_pick not in {"best", "median"}:
        raise ValueError("VAMOS_TUNER_PICK must be either 'best' or 'median'.")

    algo_space = build_nsgaii_config_space()
    param_space = algo_space.to_param_space()
    if "pop_size" not in param_space.params:
        raise KeyError("pop_size not found in NSGA-II tuning space")
    pop_param = param_space.params["pop_size"]
    if hasattr(pop_param, "low") and hasattr(pop_param, "high"):
        pop_param.low = max(int(pop_param.low), int(min_pop_size))
        if int(pop_param.high) < int(pop_param.low):
            raise ValueError(f"Invalid pop_size range after applying min={min_pop_size}: [{pop_param.low}, {pop_param.high}]")
    else:  # pragma: no cover
        raise TypeError(f"Unexpected pop_size param type: {type(pop_param)}")

    # Constrain NSGA-II to the common setting offspring_size = pop_size.
    # (This keeps the tuning space smaller and avoids pathological settings.)
    param_space.params.pop("offspring_size", None)

    instances: list[Instance] = []
    for problem in train_problems:
        n_var, n_obj = problem_dims(problem)
        instances.append(Instance(name=problem, n_var=n_var, kwargs={"n_obj": n_obj}))

    seeds = [seed0 + i for i in range(tune_n_seeds)]

    task = TuningTask(
        name="tune_nsgaii_paper_ablation",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        budget_per_run=(max(fidelity_levels) if use_multi_fidelity else tune_n_evals),
        maximize=True,
    )

    scenario_kwargs: dict[str, Any] = {
        "max_experiments": tune_max_experiments,
        "use_adaptive_budget": False,
        "verbose": True,
        "n_jobs": tune_n_jobs,
    }
    if use_multi_fidelity:
        scenario_kwargs.update(
            use_multi_fidelity=True,
            fidelity_levels=tuple(fidelity_levels),
            fidelity_warm_start=bool(warm_start),
        )
    scenario = Scenario(**scenario_kwargs)

    def _eval_score(cfg: dict[str, Any], ctx: EvalContext) -> float:
        try:
            algo_cfg = config_from_assignment("nsgaii", cfg)
            n_obj = int(ctx.instance.kwargs.get("n_obj", 2))
            selection = make_problem_selection(ctx.instance.name, n_var=ctx.instance.n_var, n_obj=n_obj)
            problem = selection.instantiate()
            res = optimize(
                problem,
                algorithm="nsgaii",
                algorithm_config=algo_cfg,
                termination=("n_eval", ctx.budget),
                seed=ctx.seed,
                engine="numpy",
            )
            return float(compute_hv(res.F, ctx.instance.name))
        except Exception:
            return 0.0

    def _run_once(*, tuner_seed: int) -> tuple[dict[str, Any], list[Any]]:
        tuner = RacingTuner(
            task=task,
            scenario=scenario,
            seed=int(tuner_seed),
            max_initial_configs=int(tune_max_initial_configs),
        )

        desc = f"Tuning (seed={tuner_seed})"
        if tune_n_jobs == 1:
            bar = ProgressBar(total=tune_max_experiments, desc=desc)

            def eval_fn(cfg: dict[str, Any], ctx: EvalContext) -> float:
                score = _eval_score(cfg, ctx)
                bar.update(1)
                return score

            try:
                return tuner.run(eval_fn)
            finally:
                bar.close()

        def eval_fn(cfg: dict[str, Any], ctx: EvalContext) -> float:
            return _eval_score(cfg, ctx)

        with joblib_progress(total=tune_max_experiments, desc=desc):
            return tuner.run(eval_fn)

    runs: list[dict[str, Any]] = []
    for rep in range(tune_repeats):
        tuner_seed = int(seed0 + rep)
        best_cfg, history = _run_once(tuner_seed=tuner_seed)
        best_score = None
        for trial in history:
            try:
                if trial.config == best_cfg:
                    best_score = float(trial.score)
                    break
            except Exception:
                continue
        if best_score is None:
            best_score = float("-inf")
        runs.append(
            {
                "repeat": int(rep),
                "tuner_seed": int(tuner_seed),
                "best_score": float(best_score),
                "best_cfg": best_cfg,
                "history": history,
            }
        )

    runs_sorted = sorted(runs, key=lambda r: r["best_score"], reverse=True)
    if tune_pick == "median":
        chosen = runs_sorted[len(runs_sorted) // 2]
    else:
        chosen = runs_sorted[0]

    best_cfg = chosen["best_cfg"]
    history = chosen["history"]

    out_cfg = Path(os.environ.get("VAMOS_TUNER_OUTPUT_JSON", str(ROOT_DIR / "experiments" / "tuned_nsgaii.json")))
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(best_cfg, indent=2, sort_keys=True), encoding="utf-8")

    out_resolved = Path(
        os.environ.get(
            "VAMOS_TUNER_OUTPUT_RESOLVED_JSON",
            str(out_cfg.with_name(f"{out_cfg.stem}_resolved.json")),
        )
    )
    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved_cfg = config_from_assignment("nsgaii", best_cfg)
    out_resolved.write_text(json.dumps(resolved_cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    out_hist_csv = Path(os.environ.get("VAMOS_TUNER_HISTORY_CSV", str(ROOT_DIR / "experiments" / "tuner_history.csv")))
    out_hist_json = out_hist_csv.with_suffix(".json")
    save_history_csv(history, param_space, out_hist_csv)
    save_history_json(history, param_space, out_hist_json)

    out_runs_csv = Path(os.environ.get("VAMOS_TUNER_RUNS_CSV", str(ROOT_DIR / "experiments" / "tuned_nsgaii_runs.csv")))
    out_runs_csv.parent.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(
        [
            {
                "repeat": r["repeat"],
                "tuner_seed": r["tuner_seed"],
                "best_score": r["best_score"],
                "selected": int(1 if r is chosen else 0),
            }
            for r in runs_sorted
        ]
    )
    runs_df.to_csv(out_runs_csv, index=False)

    return best_cfg


# =============================================================================
# Benchmark runs
# =============================================================================


def build_config(*, variant: str, seed: int, n_var: int, tuned_cfg: dict[str, Any] | None) -> NSGAIIConfig:
    base = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
    )

    if variant == "baseline":
        return base.build()
    if variant == "aos":
        return base.adaptive_operator_selection(make_aos_cfg(seed=seed, n_var=n_var)).build()
    if variant == "tuned":
        if tuned_cfg is None:
            raise ValueError("tuned variant requested but tuned_cfg is None")
        return config_from_assignment("nsgaii", tuned_cfg)
    if variant == "tuned_aos":
        if tuned_cfg is None:
            raise ValueError("tuned_aos variant requested but tuned_cfg is None")
        tuned = config_from_assignment("nsgaii", tuned_cfg)
        return replace(tuned, adaptive_operator_selection=make_aos_cfg(seed=seed, n_var=n_var))
    raise ValueError(f"Unknown variant '{variant}'")


def run_single(
    variant: str,
    problem_name: str,
    seed: int,
    *,
    n_evals: int,
    engine: str,
    tuned_cfg: dict[str, Any] | None,
    checkpoints: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_var, n_obj = problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
    algo_cfg = build_config(variant=variant, seed=seed, n_var=n_var, tuned_cfg=tuned_cfg)
    start = time.perf_counter()
    recorder = HVCheckpointRecorder(problem_name=problem_name, checkpoints=checkpoints, start_time=start) if checkpoints else None
    res = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", n_evals),
        seed=seed,
        engine=engine,
        live_viz=recorder,
    )
    elapsed = time.perf_counter() - start
    hv = compute_hv(res.F, problem_name) if res.F is not None else float("nan")

    final_row = {
        "variant": variant,
        "problem": problem_name,
        "algorithm": "NSGA-II",
        "engine": engine,
        "n_evals": n_evals,
        "seed": seed,
        "runtime_seconds": float(elapsed),
        "n_solutions": int(res.X.shape[0]) if res.X is not None else 0,
        "hypervolume": float(hv),
        "algorithm_config": json.dumps(_maybe_dataclass_to_dict(algo_cfg), sort_keys=True),
    }

    chk_rows: list[dict[str, Any]] = []
    if recorder is not None:
        for r in recorder.records():
            chk_rows.append(
                {
                    "variant": variant,
                    "problem": problem_name,
                    "engine": engine,
                    "n_evals": int(n_evals),
                    "seed": int(seed),
                    "evals": int(r["evals"]),
                    "runtime_seconds": float(r["seconds"]),
                    "hypervolume": float(r["hypervolume"]),
                }
            )

    return final_row, chk_rows


def main() -> None:
    problems = [*ZDT_PROBLEMS, *DTLZ_PROBLEMS, *WFG_PROBLEMS]
    problems_raw = os.environ.get("VAMOS_ABLATION_PROBLEMS")
    if problems_raw:
        problems = _parse_csv_list(problems_raw)
        unknown = [p for p in problems if p not in {*ZDT_PROBLEMS, *DTLZ_PROBLEMS, *WFG_PROBLEMS}]
        if unknown:
            raise ValueError(f"Unknown problems in VAMOS_ABLATION_PROBLEMS: {unknown}")

    n_evals = _as_int_env("VAMOS_N_EVALS", 100000)
    n_seeds = _as_int_env("VAMOS_N_SEEDS", 30)
    engine = _as_str_env("VAMOS_ABLATION_ENGINE", "numba")
    output_csv = Path(_as_str_env("VAMOS_ABLATION_OUTPUT_CSV", str(ROOT_DIR / "experiments" / "ablation_aos_racing_tuner.csv")))

    anytime_csv_raw = os.environ.get("VAMOS_ABLATION_ANYTIME_CSV")
    if anytime_csv_raw is None:
        anytime_csv_raw = str(ROOT_DIR / "experiments" / "ablation_aos_anytime.csv")
    anytime_csv_raw = str(anytime_csv_raw).strip()
    anytime_csv: Path | None = None
    if anytime_csv_raw and anytime_csv_raw not in {"0", "false", "False"}:
        anytime_csv = Path(anytime_csv_raw)

    checkpoints: list[int] | None = None
    if anytime_csv is not None:
        raw = _as_str_env("VAMOS_ABLATION_CHECKPOINTS", "5000,10000,20000,50000,100000")
        checkpoints = sorted(set(int(x) for x in _parse_int_list(raw) if int(x) > 0))
        if not checkpoints:
            raise ValueError("VAMOS_ABLATION_CHECKPOINTS must contain at least one positive integer checkpoint.")
        if checkpoints[-1] != n_evals:
            checkpoints.append(int(n_evals))

    variants = _parse_csv_list(_as_str_env("VAMOS_ABLATION_VARIANTS", "baseline,aos,tuned,tuned_aos"))
    variants = [v.strip().lower() for v in variants]
    for v in variants:
        if v not in {"baseline", "aos", "tuned", "tuned_aos"}:
            raise ValueError(f"Unsupported variant '{v}'. Supported: baseline,aos,tuned,tuned_aos")

    n_jobs = int(os.environ.get("VAMOS_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))

    print(f"Problems: {len(problems)}")
    print(f"Variants: {variants}")
    print(f"Evaluations per run: {n_evals:,}")
    print(f"Seeds: {n_seeds}")
    print(f"Engine: {engine}")
    print(f"Parallel workers: {n_jobs}")
    if anytime_csv is not None:
        print(f"Anytime checkpoints: {checkpoints}")
        print(f"Anytime CSV: {anytime_csv}")

    tuned_cfg: dict[str, Any] | None = None
    needs_tuned = any(v in variants for v in ("tuned", "tuned_aos"))
    if needs_tuned:
        enable = _as_int_env("VAMOS_TUNER_ENABLE", 1)
        if enable != 0:
            train_default = "zdt4,zdt6,dtlz2,dtlz6,dtlz7,wfg1,wfg9"
            train_raw = _as_str_env("VAMOS_TUNER_PROBLEMS", train_default)
            train_problems = _parse_csv_list(train_raw)
            seed0 = _as_int_env("VAMOS_TUNER_SEED0", 1)
            print(f"Tuning NSGA-II on: {train_problems}")
            t0 = time.perf_counter()
            tuned_cfg = tune_nsgaii(train_problems=train_problems, seed0=seed0)
            t1 = time.perf_counter()
            print(f"Tuning wall time: {t1 - t0:.1f}s")
        else:
            cfg_path = Path(os.environ.get("VAMOS_TUNER_OUTPUT_JSON", str(ROOT_DIR / "experiments" / "tuned_nsgaii.json")))
            tuned_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            print(f"Loaded tuned config from {cfg_path}")

    tasks = [(variant, problem, seed) for variant in variants for problem in problems for seed in range(n_seeds)]
    print(f"Total runs: {len(tasks)}")

    if n_jobs <= 1:
        bar = ProgressBar(total=len(tasks), desc="Ablation runs")
        final_rows: list[dict[str, Any]] = []
        anytime_rows: list[dict[str, Any]] = []
        for variant, problem, seed in tasks:
            final_row, chk = run_single(
                variant,
                problem,
                seed,
                n_evals=n_evals,
                engine=engine,
                tuned_cfg=tuned_cfg,
                checkpoints=checkpoints,
            )
            final_rows.append(final_row)
            anytime_rows.extend(chk)
            bar.update(1)
        bar.close()
    else:
        with joblib_progress(total=len(tasks), desc="Ablation runs"):
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single)(
                    variant,
                    problem,
                    seed,
                    n_evals=n_evals,
                    engine=engine,
                    tuned_cfg=tuned_cfg,
                    checkpoints=checkpoints,
                )
                for variant, problem, seed in tasks
            )
        final_rows = []
        anytime_rows = []
        for final_row, chk in results:
            final_rows.append(final_row)
            anytime_rows.extend(chk)

    df = pd.DataFrame(final_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")

    if anytime_csv is not None:
        anytime_csv.parent.mkdir(parents=True, exist_ok=True)
        df_any = pd.DataFrame(anytime_rows)
        df_any.to_csv(anytime_csv, index=False)
        print(f"Wrote {len(df_any)} rows to {anytime_csv}")


if __name__ == "__main__":
    main()

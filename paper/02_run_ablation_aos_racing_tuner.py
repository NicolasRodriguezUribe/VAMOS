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
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_N_JOBS: joblib workers (default: CPU count - 1)
  - VAMOS_ABLATION_ENGINE: VAMOS engine for evaluation runs (default: numba)
  - VAMOS_ABLATION_RESUME: 1/0 resume into existing CSVs (default: 0, start fresh)
  - VAMOS_ABLATION_VARIANTS: comma-separated variants to run
      (default: baseline,aos,tuned,tuned_aos)
  - VAMOS_ABLATION_OUTPUT_CSV: output CSV path
      (default: experiments/ablation_aos_racing_tuner.csv)
  - VAMOS_ABLATION_ANYTIME_CSV: optional checkpoint CSV output path
      (default: experiments/ablation_aos_anytime.csv; set empty/0 to disable)
  - VAMOS_ABLATION_CHECKPOINTS: comma-separated evaluation checkpoints for anytime HV
      (default: 5000,10000,20000,50000)
  - VAMOS_ABLATION_AOS_TRACE_CSV: optional per-generation AOS trace CSV path
      (default: disabled; set empty/0 to disable)
  - VAMOS_ABLATION_AOS_TRACE_VARIANTS: comma-separated variants to export traces for
      (default: tuned_aos)
  - VAMOS_ABLATION_AOS_TRACE_PROBLEMS: comma-separated problems to export traces for
      (default: zdt4,dtlz3,wfg9)
  - VAMOS_CHECKPOINT_INTERVAL_MIN: time-based checkpoint interval in minutes
      (default: 30; also saves at end of each seed)

Tuner controls (optional):
  - VAMOS_TUNER_ENABLE: 1/0 (default: 1 if "tuned" variant requested)
  - VAMOS_TUNER_PROBLEMS: comma-separated training problems
      (default: zdt4,zdt6,dtlz2,dtlz6,dtlz7,wfg1,wfg9)
  - VAMOS_TUNER_N_EVALS: evaluations per tuning run (default: 20000; used only if multi-fidelity is disabled)
  - VAMOS_TUNER_N_SEEDS: number of seeds per tuning block (default: 10)
  - VAMOS_TUNER_SEED0: first seed for tuning (default: 1)
  - VAMOS_TUNER_MAX_EXPERIMENTS: max config×instance×seed blocks (default: auto-estimated)
  - VAMOS_TUNER_N_JOBS: parallel jobs for tuner evaluations (default: 1)
  - VAMOS_TUNER_MAX_INITIAL_CONFIGS: number of configurations sampled per tuning run (default: auto-estimated)
  - VAMOS_TUNER_REPEATS: repeat tuning runs with different tuner seeds (default: 1)
  - VAMOS_TUNER_PICK: which repeated run to select ("best" or "median"; default: best)
  - VAMOS_TUNER_MIN_POP_SIZE: minimum pop_size considered during tuning (default: 100)
  - VAMOS_TUNER_FIXED_POP_SIZE: force pop_size to this value
      (default: 100 when external archive is disabled; set 0 to tune pop_size even without an external archive)
  - VAMOS_TUNER_USE_MULTI_FIDELITY: 1/0 enable multi-fidelity tuning (default: 1)
  - VAMOS_TUNER_FIDELITY_LEVELS: comma-separated budgets for multi-fidelity tuning
      (default: 10000,30000,50000)
  - VAMOS_TUNER_FIDELITY_WARM_START: 1/0 warm-start between fidelity levels (default: 0)
  - VAMOS_TUNER_OUTPUT_JSON: tuned config path (default: experiments/tuned_nsgaii.json)
  - VAMOS_TUNER_OUTPUT_RESOLVED_JSON: resolved NSGA-II config path
      (default: experiments/tuned_nsgaii_resolved.json)
  - VAMOS_TUNER_HISTORY_CSV: tuner history CSV (default: experiments/tuner_history.csv)
  - VAMOS_TUNER_RUNS_CSV: tuning repeats summary CSV (default: experiments/tuned_nsgaii_runs.csv)
  - VAMOS_TUNER_TOPK: export top-k configs from tuner history (default: 5)
  - VAMOS_TUNER_TOPK_JSON: top-k JSON output path (default: experiments/tuned_nsgaii_topk.json)

  AOS portfolio (tuned + AOS):
  - VAMOS_AOS_TOPK_ARMS: max number of operator arms sourced from the top-k list (default: 5)
"""

from __future__ import annotations

import json
import math
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
    from .benchmark_utils import compute_hv, _reference_hv, _reference_point
except ImportError:  # pragma: no cover
    from benchmark_utils import compute_hv, _reference_hv, _reference_point

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


def _export_topk_from_history_json(history_json: Path, out_json: Path, k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    raw = json.loads(history_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {history_json}, got {type(raw)}")

    def _score(entry: dict[str, Any]) -> float:
        try:
            return float(entry.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    sorted_entries = sorted((e for e in raw if isinstance(e, dict)), key=_score, reverse=True)
    topk: list[dict[str, Any]] = []
    for entry in sorted_entries[:k]:
        cfg = dict(entry.get("config") or {})
        if bool(cfg.get("use_external_archive", False)) and str(cfg.get("archive_type", "")).strip().lower() == "unbounded":
            cfg.pop("archive_size_factor", None)
        topk.append(
            {
                "trial_id": entry.get("trial_id"),
                "score": entry.get("score"),
                "config": cfg,
                "details": entry.get("details"),
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(topk, indent=2, sort_keys=True), encoding="utf-8")
    return topk


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


def make_aos_cfg(
    *,
    seed: int,
    n_var: int,
    operator_pool: list[dict[str, Any]] | None = None,
    problem_name: str | None = None,
) -> dict[str, Any]:
    """
    Minimal, reproducible AOS configuration:
    - enabled
    - deterministic RNG seeded per run
    - small operator portfolio (>=2 arms, otherwise AOS is a no-op)
    """
    if operator_pool is None:
        operator_pool = [
            {
                "crossover": ("sbx", {"prob": CROSSOVER_PROB, "eta": CROSSOVER_ETA}),
                "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA}),
            },
            {
                "crossover": ("pcx", {"prob": CROSSOVER_PROB, "sigma_eta": 0.1, "sigma_zeta": 0.1}),
                "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA}),
            },
            {
                "crossover": ("undx", {"prob": CROSSOVER_PROB, "zeta": 0.5, "eta": 0.35}),
                "mutation": ("gaussian", {"prob": 1.0 / n_var, "sigma": 0.1}),
            },
            {
                "crossover": ("simplex", {"prob": CROSSOVER_PROB, "epsilon": 0.5}),
                "mutation": ("uniform_reset", {"prob": 1.0 / n_var}),
            },
            {
                "crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5, "repair": "random"}),
                "mutation": ("cauchy", {"prob": 1.0 / n_var, "gamma": 0.1}),
            },
        ]
    hv_reference_point: list[float] | None = None
    hv_reference_hv: float | None = None
    if problem_name:
        try:
            ref = _reference_point(problem_name)
            hv_reference_point = [float(x) for x in ref.tolist()]
            hv_reference_hv = float(_reference_hv(problem_name))
            if hv_reference_hv <= 0.0:
                hv_reference_point = None
                hv_reference_hv = None
        except Exception:
            hv_reference_point = None
            hv_reference_hv = None
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
        "hv_reference_point": hv_reference_point,
        "hv_reference_hv": hv_reference_hv,
        "operator_pool": operator_pool,
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
    tune_n_jobs = _as_int_env("VAMOS_TUNER_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
    tune_repeats = _as_int_env("VAMOS_TUNER_REPEATS", 1)
    tune_pick = _as_str_env("VAMOS_TUNER_PICK", "best").strip().lower()
    min_pop_size = _as_int_env("VAMOS_TUNER_MIN_POP_SIZE", 100)
    fixed_pop_size_no_archive = _as_int_env("VAMOS_TUNER_FIXED_POP_SIZE", POP_SIZE)
    use_multi_fidelity = _as_int_env("VAMOS_TUNER_USE_MULTI_FIDELITY", 1) != 0
    warm_start = _as_int_env("VAMOS_TUNER_FIDELITY_WARM_START", 0) != 0

    # Optionally auto-tune the tuning budget (max_initial_configs/max_experiments)
    # based on the search space and the number of instance×seed blocks.
    max_experiments_env = os.environ.get("VAMOS_TUNER_MAX_EXPERIMENTS")
    max_initial_env = os.environ.get("VAMOS_TUNER_MAX_INITIAL_CONFIGS")
    tune_max_experiments = _as_int_env("VAMOS_TUNER_MAX_EXPERIMENTS", 0)
    tune_max_initial_configs = _as_int_env("VAMOS_TUNER_MAX_INITIAL_CONFIGS", 0)

    fidelity_levels: list[int] = []
    if use_multi_fidelity:
        raw_levels = _as_str_env("VAMOS_TUNER_FIDELITY_LEVELS", "10000,30000,50000")
        fidelity_levels = _parse_int_list(raw_levels)
        if not fidelity_levels:
            raise ValueError("VAMOS_TUNER_FIDELITY_LEVELS must contain at least one integer budget.")
        if any(b <= 0 for b in fidelity_levels):
            raise ValueError(f"Invalid fidelity budgets (must be positive): {fidelity_levels}")
        fidelity_levels = sorted(set(fidelity_levels))
        if len(fidelity_levels) < 2:
            raise ValueError("VAMOS_TUNER_FIDELITY_LEVELS must contain at least two increasing budgets.")

    algo_space = build_nsgaii_config_space()
    param_space = algo_space.to_param_space()

    def _choices_count(name: str) -> int:
        param = param_space.params.get(name)
        choices = getattr(param, "choices", None)
        return int(len(choices)) if choices is not None else 1

    if max_initial_env is None or tune_max_initial_configs <= 0:
        n_cross = _choices_count("crossover")
        n_mut = _choices_count("mutation")
        n_offspring = _choices_count("offspring_ratio")
        n_repair = _choices_count("repair")
        n_init = _choices_count("initializer")
        n_operator_combos = max(1, int(n_cross) * int(n_mut) * int(n_offspring) * int(n_repair) * int(n_init))
        tune_max_initial_configs = int(max(60, min(250, round(0.6 * n_operator_combos))))

    if tune_max_experiments <= 0:
        n_blocks = max(1, len(train_problems) * tune_n_seeds)
        if use_multi_fidelity:
            promo = 0.3
            mf_factor = sum(promo**i for i in range(len(fidelity_levels)))
        else:
            mf_factor = 1.0
        min_budget = float(tune_max_initial_configs) * float(n_blocks) * float(mf_factor)
        suggested = int(math.ceil(min_budget * 1.2))
        tune_max_experiments = suggested

        print(
            "[tuner] Estimated minimum max_experiments ≈ "
            f"{int(math.ceil(min_budget))} (configs={tune_max_initial_configs}, blocks={n_blocks}, factor={mf_factor:.2f}); "
            f"using {tune_max_experiments}."
        )

    if tune_max_initial_configs < 1:
        raise ValueError("VAMOS_TUNER_MAX_INITIAL_CONFIGS must be >= 1.")
    if tune_max_experiments < 1:
        raise ValueError("VAMOS_TUNER_MAX_EXPERIMENTS must be >= 1.")
    if tune_repeats < 1:
        raise ValueError("VAMOS_TUNER_REPEATS must be >= 1.")
    if tune_pick not in {"best", "median"}:
        raise ValueError("VAMOS_TUNER_PICK must be either 'best' or 'median'.")

    if "pop_size" not in param_space.params:
        raise KeyError("pop_size not found in NSGA-II tuning space")
    pop_param = param_space.params["pop_size"]
    if hasattr(pop_param, "low") and hasattr(pop_param, "high"):
        pop_param.low = max(int(pop_param.low), int(min_pop_size))
        if int(pop_param.high) < int(pop_param.low):
            raise ValueError(f"Invalid pop_size range after applying min={min_pop_size}: [{pop_param.low}, {pop_param.high}]")
        if fixed_pop_size_no_archive > 0:
            if fixed_pop_size_no_archive < int(min_pop_size):
                raise ValueError(
                    f"VAMOS_TUNER_FIXED_POP_SIZE={fixed_pop_size_no_archive} must be >= VAMOS_TUNER_MIN_POP_SIZE={min_pop_size}."
                )
            if not (int(pop_param.low) <= int(fixed_pop_size_no_archive) <= int(pop_param.high)):
                raise ValueError(
                    f"VAMOS_TUNER_FIXED_POP_SIZE={fixed_pop_size_no_archive} must be within the tuned pop_size range [{pop_param.low}, {pop_param.high}]."
                )
    else:  # pragma: no cover
        raise TypeError(f"Unexpected pop_size param type: {type(pop_param)}")

    # Offspring size is controlled via offspring_ratio (<= pop_size) in the tuning space.

    class _PopSizePolicySampler:
        def __init__(self, space: Any, *, fixed_pop_size_no_archive: int) -> None:
            self._space = space
            self._fixed = int(fixed_pop_size_no_archive)

        def sample(self, rng: Any) -> dict[str, Any]:
            cfg = self._space.sample(rng)
            if self._fixed > 0 and not bool(cfg.get("use_external_archive", False)):
                cfg["pop_size"] = int(self._fixed)
            return cfg

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
        sampler = _PopSizePolicySampler(param_space, fixed_pop_size_no_archive=fixed_pop_size_no_archive)
        tuner = RacingTuner(
            task=task,
            scenario=scenario,
            seed=int(tuner_seed),
            max_initial_configs=int(tune_max_initial_configs),
            sampler=sampler,
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
    if fixed_pop_size_no_archive > 0 and not bool(best_cfg.get("use_external_archive", False)):
        best_cfg["pop_size"] = int(fixed_pop_size_no_archive)
    if bool(best_cfg.get("use_external_archive", False)) and str(best_cfg.get("archive_type", "")).strip().lower() == "unbounded":
        # archive_size_factor is ignored for unbounded archives; keep JSON minimal.
        best_cfg.pop("archive_size_factor", None)
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
    resolved_dict = resolved_cfg.to_dict()
    if resolved_dict.get("archive_type") is None:
        resolved_dict.pop("archive_type", None)
    out_resolved.write_text(json.dumps(resolved_dict, indent=2, sort_keys=True), encoding="utf-8")

    out_hist_csv = Path(os.environ.get("VAMOS_TUNER_HISTORY_CSV", str(ROOT_DIR / "experiments" / "tuner_history.csv")))
    out_hist_json = out_hist_csv.with_suffix(".json")
    save_history_csv(history, param_space, out_hist_csv)
    save_history_json(history, param_space, out_hist_json)

    topk = _as_int_env("VAMOS_TUNER_TOPK", 5)
    out_topk_json = Path(os.environ.get("VAMOS_TUNER_TOPK_JSON", str(out_cfg.with_name(f"{out_cfg.stem}_topk.json"))))
    if topk > 0:
        _export_topk_from_history_json(out_hist_json, out_topk_json, topk)

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


def build_config(*, variant: str, seed: int, n_var: int, tuned_cfg: dict[str, Any] | None, problem_name: str) -> NSGAIIConfig:
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
        return base.adaptive_operator_selection(make_aos_cfg(seed=seed, n_var=n_var, problem_name=problem_name)).build()
    if variant == "tuned":
        if tuned_cfg is None:
            raise ValueError("tuned variant requested but tuned_cfg is None")
        return config_from_assignment("nsgaii", tuned_cfg)
    if variant == "tuned_aos":
        if tuned_cfg is None:
            raise ValueError("tuned_aos variant requested but tuned_cfg is None")
        tuned = config_from_assignment("nsgaii", tuned_cfg)

        def _sig(entry: dict[str, Any]) -> str:
            return json.dumps(entry, sort_keys=True)

        operator_pool: list[dict[str, Any]] = [{"crossover": tuned.crossover, "mutation": tuned.mutation}]
        seen = {_sig(operator_pool[0])}

        topk_path = Path(os.environ.get("VAMOS_TUNER_TOPK_JSON", str(ROOT_DIR / "experiments" / "tuned_nsgaii_topk.json")))
        max_arms = _as_int_env("VAMOS_AOS_TOPK_ARMS", 5)
        if max_arms > 1 and topk_path.exists():
            try:
                topk_raw = json.loads(topk_path.read_text(encoding="utf-8"))
                if isinstance(topk_raw, list):
                    for item in topk_raw:
                        if not isinstance(item, dict):
                            continue
                        cfg = item.get("config")
                        if not isinstance(cfg, dict):
                            continue
                        resolved = config_from_assignment("nsgaii", cfg)
                        entry = {"crossover": resolved.crossover, "mutation": resolved.mutation}
                        sig = _sig(entry)
                        if sig in seen:
                            continue
                        operator_pool.append(entry)
                        seen.add(sig)
                        if len(operator_pool) >= max_arms:
                            break
            except Exception:
                pass

        if len(operator_pool) < 2:
            operator_pool.append({"crossover": tuned.crossover, "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA})})

        return replace(
            tuned,
            adaptive_operator_selection=make_aos_cfg(seed=seed, n_var=n_var, operator_pool=operator_pool, problem_name=problem_name),
        )
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
    capture_aos_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    n_var, n_obj = problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
    algo_cfg = build_config(variant=variant, seed=seed, n_var=n_var, tuned_cfg=tuned_cfg, problem_name=problem_name)
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

    trace_rows: list[dict[str, Any]] = []
    if capture_aos_trace:
        try:
            aos_payload = res.data.get("aos")
            if isinstance(aos_payload, dict):
                raw = aos_payload.get("trace_rows")
                if isinstance(raw, list):
                    pop_size = int(getattr(algo_cfg, "pop_size", POP_SIZE))
                    offspring_size = int(getattr(algo_cfg, "offspring_size", None) or pop_size)
                    for row in raw:
                        if not isinstance(row, dict):
                            continue
                        step = int(row.get("step", 0))
                        batch_size = int(row.get("batch_size") or offspring_size)
                        evals_after = pop_size + int((step + 1) * batch_size)
                        trace_rows.append(
                            {
                                "variant": variant,
                                "problem": str(problem_name).lower(),
                                "engine": str(engine).lower(),
                                "n_evals": int(n_evals),
                                "seed": int(seed),
                                "step": step,
                                "evals": int(evals_after),
                                "op_id": row.get("op_id"),
                                "op_name": row.get("op_name"),
                                "batch_size": int(batch_size),
                                "reward": float(row.get("reward", 0.0)),
                                "reward_survival": float(row.get("reward_survival", 0.0)),
                                "reward_nd_insertions": float(row.get("reward_nd_insertions", 0.0)),
                                "reward_hv_delta": float(row.get("reward_hv_delta", 0.0)),
                            }
                        )
        except Exception:
            trace_rows = []

    return final_row, chk_rows, trace_rows


def main() -> None:
    problems = [*ZDT_PROBLEMS, *DTLZ_PROBLEMS, *WFG_PROBLEMS]
    problems_raw = os.environ.get("VAMOS_ABLATION_PROBLEMS")
    if problems_raw:
        problems = _parse_csv_list(problems_raw)
        unknown = [p for p in problems if p not in {*ZDT_PROBLEMS, *DTLZ_PROBLEMS, *WFG_PROBLEMS}]
        if unknown:
            raise ValueError(f"Unknown problems in VAMOS_ABLATION_PROBLEMS: {unknown}")

    n_evals = _as_int_env("VAMOS_N_EVALS", 50000)
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

    aos_trace_csv_raw = str(os.environ.get("VAMOS_ABLATION_AOS_TRACE_CSV", "")).strip()
    aos_trace_csv: Path | None = None
    if aos_trace_csv_raw and aos_trace_csv_raw not in {"0", "false", "False"}:
        aos_trace_csv = Path(aos_trace_csv_raw)
    trace_variants = set(v.strip().lower() for v in _parse_csv_list(_as_str_env("VAMOS_ABLATION_AOS_TRACE_VARIANTS", "tuned_aos")))
    trace_problems = set(p.strip().lower() for p in _parse_csv_list(_as_str_env("VAMOS_ABLATION_AOS_TRACE_PROBLEMS", "zdt4,dtlz3,wfg9")))

    checkpoints: list[int] | None = None
    if anytime_csv is not None:
        raw = _as_str_env("VAMOS_ABLATION_CHECKPOINTS", "5000,10000,20000,50000")
        checkpoints = sorted(set(int(x) for x in _parse_int_list(raw) if 0 < int(x) <= n_evals))
        if not checkpoints:
            raise ValueError("VAMOS_ABLATION_CHECKPOINTS must contain at least one positive integer checkpoint.")
        if n_evals not in checkpoints:
            checkpoints.append(int(n_evals))
        checkpoints = sorted(set(checkpoints))

    variants = _parse_csv_list(_as_str_env("VAMOS_ABLATION_VARIANTS", "baseline,aos,tuned,tuned_aos"))
    variants = [v.strip().lower() for v in variants]
    for v in variants:
        if v not in {"baseline", "aos", "tuned", "tuned_aos"}:
            raise ValueError(f"Unsupported variant '{v}'. Supported: baseline,aos,tuned,tuned_aos")

    # joblib supports negative n_jobs (e.g., -1 = all cores). Only n_jobs==1 is truly sequential.
    n_jobs = int(os.environ.get("VAMOS_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
    if n_jobs == 0:
        raise ValueError("VAMOS_N_JOBS cannot be 0 (joblib expects 1, -1, or another non-zero integer).")

    print(f"Problems: {len(problems)}")
    print(f"Variants: {variants}")
    print(f"Evaluations per run: {n_evals:,}")
    print(f"Seeds: {n_seeds}")
    print(f"Engine: {engine}")
    print(f"Parallel workers: {n_jobs}")
    if anytime_csv is not None:
        print(f"Anytime checkpoints: {checkpoints}")
        print(f"Anytime CSV: {anytime_csv}")
    if aos_trace_csv is not None:
        print(f"AOS trace CSV: {aos_trace_csv}")
        print(f"AOS trace variants: {sorted(trace_variants)}")
        print(f"AOS trace problems: {sorted(trace_problems)}")

    resume = int(os.environ.get("VAMOS_ABLATION_RESUME", "0")) != 0

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
            topk = _as_int_env("VAMOS_TUNER_TOPK", 5)
            if topk > 0:
                hist_csv = Path(os.environ.get("VAMOS_TUNER_HISTORY_CSV", str(ROOT_DIR / "experiments" / "tuner_history.csv")))
                hist_json = hist_csv.with_suffix(".json")
                out_topk_json = Path(os.environ.get("VAMOS_TUNER_TOPK_JSON", str(cfg_path.with_name(f"{cfg_path.stem}_topk.json"))))
                if not out_topk_json.exists() and hist_json.exists():
                    _export_topk_from_history_json(hist_json, out_topk_json, topk)

    tasks = [(variant, problem, seed) for variant in variants for problem in problems for seed in range(n_seeds)]

    # Incremental write: append after each completed run to avoid losing progress
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if anytime_csv is not None:
        anytime_csv.parent.mkdir(parents=True, exist_ok=True)
    if aos_trace_csv is not None:
        aos_trace_csv.parent.mkdir(parents=True, exist_ok=True)

    if not resume:
        if output_csv.exists():
            output_csv.unlink()
        if anytime_csv is not None and anytime_csv.exists():
            anytime_csv.unlink()
        if aos_trace_csv is not None and aos_trace_csv.exists():
            aos_trace_csv.unlink()

    written_final = 0
    written_anytime = 0
    written_trace = 0

    done: set[tuple[str, str, int]] = set()
    if resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        written_final = int(len(existing_df))
        if {"variant", "problem", "seed"}.issubset(existing_df.columns):
            if "n_evals" in existing_df.columns:
                existing_df["n_evals"] = existing_df["n_evals"].astype(int)
                existing_df = existing_df[existing_df["n_evals"] == n_evals]
            if "engine" in existing_df.columns:
                existing_df["engine"] = existing_df["engine"].astype(str).str.strip().str.lower()
                existing_df = existing_df[existing_df["engine"] == str(engine).strip().lower()]
            existing_df["variant"] = existing_df["variant"].astype(str).str.strip().str.lower()
            existing_df["problem"] = existing_df["problem"].astype(str).str.strip().str.lower()
            existing_df["seed"] = existing_df["seed"].astype(int)
            done = set(zip(existing_df["variant"], existing_df["problem"], existing_df["seed"]))
            if done:
                print(f"Resume enabled: skipping {len(done)} completed runs from {output_csv}")

    if resume and anytime_csv is not None and anytime_csv.exists():
        try:
            existing_any = pd.read_csv(anytime_csv)
            written_anytime = int(len(existing_any))
        except Exception:
            written_anytime = 0

    if resume and aos_trace_csv is not None and aos_trace_csv.exists():
        try:
            existing_trace = pd.read_csv(aos_trace_csv)
            written_trace = int(len(existing_trace))
        except Exception:
            written_trace = 0

    if done:
        tasks = [t for t in tasks if t not in done]

    print(f"Total runs: {len(tasks)}")
    if resume and not tasks:
        print("Nothing to do: all requested runs already exist in the output CSV.")
        return

    # Time-based checkpointing (default: every 30 minutes)
    checkpoint_interval_sec = float(_as_int_env("VAMOS_CHECKPOINT_INTERVAL_MIN", 30)) * 60.0
    last_checkpoint_time = time.perf_counter()
    pending_final: list[dict[str, Any]] = []
    pending_anytime: list[dict[str, Any]] = []
    pending_trace: list[dict[str, Any]] = []

    def _flush_pending() -> None:
        nonlocal written_final, written_anytime, written_trace, pending_final, pending_anytime, pending_trace, last_checkpoint_time
        if pending_final:
            df_batch = pd.DataFrame(pending_final)
            df_batch.to_csv(output_csv, mode="a", header=(written_final == 0), index=False)
            written_final += len(pending_final)
            pending_final = []
        if anytime_csv is not None and pending_anytime:
            df_batch = pd.DataFrame(pending_anytime)
            df_batch.to_csv(anytime_csv, mode="a", header=(written_anytime == 0), index=False)
            written_anytime += len(pending_anytime)
            pending_anytime = []
        if aos_trace_csv is not None and pending_trace:
            df_batch = pd.DataFrame(pending_trace)
            df_batch.to_csv(aos_trace_csv, mode="a", header=(written_trace == 0), index=False)
            written_trace += len(pending_trace)
            pending_trace = []
        last_checkpoint_time = time.perf_counter()

    def _append_results(
        final_row: dict[str, Any],
        chk_rows: list[dict[str, Any]],
        trace_rows: list[dict[str, Any]],
        *,
        force_flush: bool = False,
    ) -> None:
        nonlocal last_checkpoint_time
        pending_final.append(final_row)
        pending_anytime.extend(chk_rows)
        pending_trace.extend(trace_rows)

        elapsed = time.perf_counter() - last_checkpoint_time
        # Flush if: forced, or time elapsed >= interval
        if force_flush or elapsed >= checkpoint_interval_sec:
            _flush_pending()
            if not force_flush:
                print(f"[checkpoint] Saved {written_final} rows after {elapsed/60:.1f} min")

    if n_jobs == 1:
        bar = ProgressBar(total=len(tasks), desc="Ablation runs")
        for i, (variant, problem, seed) in enumerate(tasks):
            capture = aos_trace_csv is not None and variant in trace_variants and str(problem).lower() in trace_problems
            final_row, chk, trace_rows = run_single(
                variant,
                problem,
                seed,
                n_evals=n_evals,
                engine=engine,
                tuned_cfg=tuned_cfg,
                checkpoints=checkpoints,
                capture_aos_trace=capture,
            )
            # Force flush at end of each seed (seed changes or last task)
            is_last = (i == len(tasks) - 1)
            next_seed = tasks[i + 1][2] if not is_last else None
            force = is_last or (next_seed != seed)
            _append_results(final_row, chk, trace_rows, force_flush=force)
            bar.update(1)
        bar.close()
    else:
        # For parallel execution, consume results as they complete so we can
        # checkpoint progress to disk and avoid mixing runs on subsequent calls.
        with joblib_progress(total=len(tasks), desc="Ablation runs"):
            parallel = Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
            for final_row, chk, trace_rows in parallel(
                delayed(run_single)(
                    variant,
                    problem,
                    seed,
                    n_evals=n_evals,
                    engine=engine,
                    tuned_cfg=tuned_cfg,
                    checkpoints=checkpoints,
                    capture_aos_trace=(aos_trace_csv is not None and variant in trace_variants and str(problem).lower() in trace_problems),
                )
                for variant, problem, seed in tasks
            ):
                _append_results(final_row, chk, trace_rows, force_flush=False)
        _flush_pending()  # Final flush

    print(f"Wrote {written_final} total rows to {output_csv}")
    if anytime_csv is not None:
        print(f"Wrote {written_anytime} total rows to {anytime_csv}")
    if aos_trace_csv is not None:
        print(f"Wrote {written_trace} total rows to {aos_trace_csv}")


if __name__ == "__main__":
    main()

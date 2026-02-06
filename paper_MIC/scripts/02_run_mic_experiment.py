"""
MIC Paper Experiment: AOS on Real-World Engineering Problems
============================================================
Comparisons on 21 continuous engineering surrogates from the Tanabe--Ishibuchi
RE and Zapotecas-Martínez RWA benchmark suites.

Common variants:
  (1) fixed NSGA-II (SBX + PM) — ``baseline``
  (2) random arm from the same 5-operator pool — ``random``
  (3) NSGA-II + AOS (ε-greedy, same pool) — ``aos``

Factorial (2x2) variants for causal analysis:
  (4) fixed NSGA-II + external archive — ``baseline_archive``
  (5) NSGA-II + AOS + external archive — ``aos_archive``

Usage
-----
  # Step 1: Generate reference fronts (one-time, ~hours)
  python paper_MIC/scripts/02_run_mic_experiment.py --generate-ref-fronts

  # Step 2: Run the main experiment
  python paper_MIC/scripts/02_run_mic_experiment.py

Environment variables (main experiment)
---------------------------------------
  VAMOS_MIC_N_EVALS        Evaluations per run (default: 50000)
  VAMOS_MIC_N_SEEDS        Seeds per (variant, problem) (default: 30)
  VAMOS_MIC_N_JOBS         Parallel workers (default: CPU count − 1)
  VAMOS_MIC_ENGINE         Engine backend (default: numba)
  VAMOS_MIC_RESUME         1/0 resume into existing CSVs (default: 0)
  VAMOS_MIC_VARIANTS       Comma-separated variants (default: baseline,random,aos)
  VAMOS_MIC_AOS_METHOD     Policy for variant=aos (epsilon_greedy|ucb|sliding_ucb|exp3|thompson_sampling)
  VAMOS_MIC_AOS_EPSILON    Epsilon for epsilon-greedy (default: 0.05)
  VAMOS_MIC_AOS_C          UCB exploration coefficient (default: 1.0)
  VAMOS_MIC_AOS_GAMMA      EXP3 gamma (default: 0.2)
  VAMOS_MIC_AOS_MIN_USAGE  Warm-up pulls per arm (default: 0)
  VAMOS_MIC_AOS_WINDOW_SIZE Sliding-window size for sliding_ucb / thompson_sampling (default: 50)
  VAMOS_MIC_AOS_FLOOR_PROB Extra random-selection floor probability in [0,1] (default: 0.0)
  VAMOS_MIC_AOS_DISABLE_MANYOBJ 1/0: for variant=aos, fallback to baseline on >=5 objectives (default: 0)
  VAMOS_MIC_ARCHIVE_SIZE  External archive size for *_archive variants (default: pop_size=100)
  VAMOS_MIC_ARCHIVE_TYPE  Archive pruning policy for bounded archives (default: hypervolume)
  VAMOS_MIC_ARCHIVE_UNBOUNDED 1/0: use unbounded archive for *_archive variants (default: 0)
  VAMOS_MIC_OUTPUT_CSV     Output CSV path
  VAMOS_MIC_ANYTIME_CSV    Anytime-checkpoint CSV (set 0 to disable)
  VAMOS_MIC_TRACE_CSV      AOS trace CSV (set 0 to disable)
  VAMOS_MIC_CHECKPOINTS    Comma-separated eval checkpoints (default: 5000,10000,20000,50000)
  VAMOS_MIC_TRACE_VARIANTS Variants to capture traces for (default: aos)
  VAMOS_MIC_TRACE_PROBLEMS Problems to capture traces for (default: re37,rwa2,rwa9)
  VAMOS_CHECKPOINT_INTERVAL_MIN  Flush interval in minutes (default: 30)

Reference-front generation (--generate-ref-fronts)
--------------------------------------------------
  VAMOS_MIC_REF_N_EVALS    Budget per reference run (default: 200000)
  VAMOS_MIC_REF_N_SEEDS    Seeds per reference run (default: 50)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "paper"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.metrics.pareto import pareto_filter
from vamos.foundation.problem.registry import make_problem_selection

from progress_utils import ProgressBar, joblib_progress

# =============================================================================
# Problem suite: 21 continuous engineering surrogates
# =============================================================================

# 11 continuous Tanabe-Ishibuchi RE problems
RE_PROBLEMS = [
    "re21", "re24",                               # 2-obj
    "re31", "re32", "re33", "re34", "re37",        # 3-obj
    "re41", "re42",                                # 4-obj
    "re61",                                        # 6-obj
    "re91",                                        # 9-obj
]

# 10 continuous Zapotecas-Martínez RWA problems
RWA_PROBLEMS = [
    "rwa1",                                        # 2-obj
    "rwa2", "rwa3", "rwa4", "rwa5", "rwa6", "rwa7",  # 3-obj
    "rwa8",                                        # 4-obj
    "rwa9",                                        # 5-obj
    "rwa10",                                       # 7-obj
]

ALL_PROBLEMS = [*RE_PROBLEMS, *RWA_PROBLEMS]

# Objective counts (from the registry specs)
_N_OBJ: dict[str, int] = {
    "re21": 2, "re24": 2,
    "re31": 3, "re32": 3, "re33": 3, "re34": 3, "re37": 3,
    "re41": 4, "re42": 4,
    "re61": 6, "re91": 9,
    "rwa1": 2,
    "rwa2": 3, "rwa3": 3, "rwa4": 3, "rwa5": 3, "rwa6": 3, "rwa7": 3,
    "rwa8": 4,
    "rwa9": 5,
    "rwa10": 7,
}

# Objective-count group label (for the paper tables)
def obj_group(problem: str) -> str:
    m = _N_OBJ.get(problem, 0)
    if m <= 2:
        return "2-obj"
    if m == 3:
        return "3-obj"
    if m == 4:
        return "4-obj"
    return "many-obj"

OBJ_GROUP_ORDER = ("2-obj", "3-obj", "4-obj", "many-obj")

# =============================================================================
# Shared defaults
# =============================================================================
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

REFERENCE_FRONTS_DIR = ROOT_DIR / "data" / "reference_fronts"
REF_EPS = 1e-6



# =============================================================================
# Operator portfolio (same 5 arms as the original paper)
# =============================================================================

def _operator_pool(n_var: int) -> list[dict[str, Any]]:
    mut_prob = 1.0 / max(n_var, 1)
    return [
        {
            "crossover": ("sbx", {"prob": CROSSOVER_PROB, "eta": CROSSOVER_ETA}),
            "mutation": ("pm", {"prob": mut_prob, "eta": MUTATION_ETA}),
        },
        {
            "crossover": ("pcx", {"prob": CROSSOVER_PROB, "sigma_eta": 0.1, "sigma_zeta": 0.1}),
            "mutation": ("pm", {"prob": mut_prob, "eta": MUTATION_ETA}),
        },
        {
            "crossover": ("undx", {"prob": CROSSOVER_PROB, "zeta": 0.5, "eta": 0.35}),
            "mutation": ("gaussian", {"prob": mut_prob, "sigma": 0.1}),
        },
        {
            "crossover": ("simplex", {"prob": CROSSOVER_PROB, "epsilon": 0.5}),
            "mutation": ("uniform_reset", {"prob": mut_prob}),
        },
        {
            "crossover": ("blx_alpha", {"prob": 0.9, "alpha": 0.5, "repair": "random"}),
            "mutation": ("cauchy", {"prob": mut_prob, "gamma": 0.1}),
        },
    ]

# =============================================================================
# HV helpers (reuse reference fronts from data/reference_fronts/)
# =============================================================================
#
# Exact hypervolume is exponential in the number of objectives.  For problems
# with > 4 objectives the reference fronts can contain 1000+ points and exact
# HV becomes infeasible.  We cap the front size used for HV computation:
#   - <= 4 objectives: use the full front (exact HV is fast).
#   - 5 objectives:    subsample to at most 200 points.
#   - 6+ objectives:   subsample to at most 100 points.
# The subsampling preserves the extreme points and fills the rest with a
# random subset, which gives a stable approximation for normalisation.

_MAX_HV_POINTS = {5: 200, 6: 100, 7: 80, 8: 60, 9: 50}
_HV_CACHE: dict[str, float] = {}
_REF_POINT_CACHE: dict[str, np.ndarray] = {}


def _subsample_front(front: np.ndarray, max_points: int) -> np.ndarray:
    """Keep extreme points plus a random subset to stay within *max_points*."""
    if front.shape[0] <= max_points:
        return front
    n_obj = front.shape[1]
    # Always keep per-objective extremes (min and max)
    extreme_idx: set[int] = set()
    for m in range(n_obj):
        extreme_idx.add(int(np.argmin(front[:, m])))
        extreme_idx.add(int(np.argmax(front[:, m])))
    extreme_idx_list = sorted(extreme_idx)
    remaining = max_points - len(extreme_idx_list)
    if remaining <= 0:
        return front[extreme_idx_list[:max_points]]
    other_idx = np.array([i for i in range(front.shape[0]) if i not in extreme_idx])
    rng = np.random.RandomState(42)
    chosen = rng.choice(other_idx, size=min(remaining, len(other_idx)), replace=False)
    idx = np.sort(np.concatenate([np.array(extreme_idx_list), chosen]))
    return front[idx]


def _load_reference_front(problem_name: str) -> np.ndarray:
    name = problem_name.lower()
    path = REFERENCE_FRONTS_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference front for '{name}': {path}")
    return np.loadtxt(path, delimiter=",")


def _reference_point(problem_name: str) -> np.ndarray:
    name = problem_name.lower()
    if name in _REF_POINT_CACHE:
        return _REF_POINT_CACHE[name]
    front = _load_reference_front(name)
    ref = front.max(axis=0) + REF_EPS
    _REF_POINT_CACHE[name] = ref
    return ref


def _hv_safe(front: np.ndarray, ref: np.ndarray) -> float:
    """Compute HV, subsampling large fronts for many-objective problems."""
    n_obj = front.shape[1] if front.ndim == 2 else 0
    if n_obj >= 5:
        max_pts = _MAX_HV_POINTS.get(n_obj, 50)
        front = _subsample_front(front, max_pts)
    return hypervolume(front, ref, allow_ref_expand=False) if front.size else 0.0


def _reference_hv(problem_name: str) -> float:
    name = problem_name.lower()
    if name in _HV_CACHE:
        return _HV_CACHE[name]
    front = _load_reference_front(name)
    ref = _reference_point(name)
    front = front[np.all(front <= ref, axis=1)]
    hv = _hv_safe(front, ref)
    _HV_CACHE[name] = hv
    return hv


def compute_hv(F: Any, problem_name: str) -> float:
    """Compute normalized HV using a pre-built reference front."""
    if F is None:
        return float("nan")
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.size == 0:
        return 0.0
    front = pareto_filter(F_arr)
    if front is None or front.size == 0:
        return 0.0
    ref = _reference_point(problem_name)
    front = front[np.all(front <= ref, axis=1)]
    if front.size == 0:
        return 0.0
    hv = _hv_safe(front, ref)
    hv_ref = _reference_hv(problem_name)
    return hv / hv_ref if hv_ref > 0.0 else 0.0

# =============================================================================
# AOS / variant configuration
# =============================================================================

def _hv_reward_refs(problem_name: str) -> tuple[list[float] | None, float | None]:
    """Return (hv_reference_point, hv_reference_hv) or (None, None) for many-obj."""
    n_obj = _N_OBJ.get(problem_name.lower(), 2)
    if n_obj >= 5:
        return None, None
    try:
        ref = _reference_point(problem_name)
        hv_reference_point = [float(x) for x in ref.tolist()]
        hv_reference_hv = float(_reference_hv(problem_name))
        if hv_reference_hv <= 0.0:
            return None, None
        return hv_reference_point, hv_reference_hv
    except Exception:
        return None, None


def _make_aos_cfg(
    *,
    seed: int,
    n_var: int,
    problem_name: str,
    epsilon: float = 0.05,
    method: str = "epsilon_greedy",
    min_usage: int = 0,
    c: float = 1.0,
    gamma: float = 0.2,
    window_size: int = 0,
    floor_prob: float = 0.0,
) -> dict[str, Any]:
    """Build AOS config dict for any AOS variant."""
    hv_ref_pt, hv_ref_hv = _hv_reward_refs(problem_name)
    n_obj = _N_OBJ.get(problem_name.lower(), 2)

    # For many-obj without HV feedback, shift weight to survival + nd_insertions
    if hv_ref_pt is None:
        weights = {"survival": 0.50, "nd_insertions": 0.50, "hv_delta": 0.00}
    else:
        weights = {"survival": 0.40, "nd_insertions": 0.40, "hv_delta": 0.20}

    return {
        "enabled": True,
        "method": method,
        "epsilon": float(epsilon),
        "c": float(c),
        "gamma": float(gamma),
        "min_usage": int(min_usage),
        "window_size": int(window_size),
        "rng_seed": int(seed),
        "floor_prob": float(floor_prob),
        "reward_scope": "combined",
        "reward_weights": weights,
        "hv_reference_point": hv_ref_pt,
        "hv_reference_hv": hv_ref_hv,
        "operator_pool": _operator_pool(n_var),
    }


# ---- Variant definitions ----
@dataclass(frozen=True)
class VariantSpec:
    aos_kwargs: dict[str, Any] | None
    use_archive: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "baseline": VariantSpec(aos_kwargs=None, use_archive=False),
    "random": VariantSpec(aos_kwargs=dict(method="epsilon_greedy", epsilon=1.0, min_usage=0), use_archive=False),
    # Original AOS (weak exploration)
    "aos": VariantSpec(aos_kwargs=dict(method="epsilon_greedy", epsilon=0.05, min_usage=0), use_archive=False),
    # 2x2 factorial cells: archive off/on x AOS off/on
    "baseline_archive": VariantSpec(aos_kwargs=None, use_archive=True),
    "aos_archive": VariantSpec(aos_kwargs=dict(method="epsilon_greedy", epsilon=0.05, min_usage=0), use_archive=True),
    # ---- Pilot variants ----
    "aos_eps15": VariantSpec(aos_kwargs=dict(method="epsilon_greedy", epsilon=0.15, min_usage=3), use_archive=False),
    "aos_swucb": VariantSpec(aos_kwargs=dict(method="sliding_ucb", c=1.0, min_usage=3, window_size=50), use_archive=False),
    "aos_ts": VariantSpec(aos_kwargs=dict(method="thompson_sampling", min_usage=3, window_size=50), use_archive=False),
}

VALID_VARIANTS = set(VARIANT_SPECS.keys())
VALID_AOS_METHODS = {"epsilon_greedy", "ucb", "sliding_ucb", "exp3", "thompson_sampling"}


@dataclass(frozen=True)
class AOSRuntimeOptions:
    """Runtime overrides for the `aos` variant via environment variables."""

    method: str = "epsilon_greedy"
    epsilon: float = 0.05
    c: float = 1.0
    gamma: float = 0.2
    min_usage: int = 0
    window_size: int = 50
    floor_prob: float = 0.0
    disable_manyobj: bool = False


@dataclass(frozen=True)
class ArchiveRuntimeOptions:
    """Runtime options for external-archive variants."""

    size: int = POP_SIZE
    archive_type: str = "hypervolume"
    unbounded: bool = False


def build_config(
    *,
    variant: str,
    seed: int,
    n_var: int,
    problem_name: str,
    aos_options: AOSRuntimeOptions | None = None,
    archive_options: ArchiveRuntimeOptions | None = None,
) -> NSGAIIConfig:
    """Build NSGAIIConfig for any supported variant."""
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant '{variant}'. Supported: {sorted(VALID_VARIANTS)}")
    spec = VARIANT_SPECS[variant]
    base = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / max(n_var, 1), eta=MUTATION_ETA)
        .selection("tournament")
    )
    if spec.use_archive:
        opts = archive_options or ArchiveRuntimeOptions()
        archive_size = int(opts.size)
        if archive_size <= 0:
            raise ValueError("Archive size must be positive for *_archive variants.")
        base = base.archive(archive_size, unbounded=bool(opts.unbounded))
        if not bool(opts.unbounded):
            base = base.archive_type(opts.archive_type).result_mode("external_archive")
    kwargs = spec.aos_kwargs
    if kwargs is None:
        return base.build()
    # Allow runtime policy control for the canonical "aos" variant.
    if variant in {"aos", "aos_archive"} and aos_options is not None:
        kwargs = {
            "method": aos_options.method,
            "epsilon": aos_options.epsilon,
            "c": aos_options.c,
            "gamma": aos_options.gamma,
            "min_usage": aos_options.min_usage,
            "window_size": aos_options.window_size,
            "floor_prob": aos_options.floor_prob,
        }
    # Optional safeguard: disable AOS on many-objective instances.
    if (
        variant != "random"
        and aos_options is not None
        and aos_options.disable_manyobj
        and _N_OBJ.get(problem_name, 0) >= 5
    ):
        return base.build()
    aos_cfg = _make_aos_cfg(seed=seed, n_var=n_var, problem_name=problem_name, **kwargs)
    return base.adaptive_operator_selection(aos_cfg).build()

# =============================================================================
# Checkpoint recorder
# =============================================================================

class HVCheckpointRecorder:
    """Record normalized HV at evaluation checkpoints."""

    def __init__(self, *, problem_name: str, checkpoints: list[int], start_time: float):
        self.problem_name = str(problem_name).lower()
        self.checkpoints = sorted(set(int(c) for c in checkpoints))
        self._start_time = float(start_time)
        self._next_idx = 0
        self._records: list[dict[str, float]] = []
        self._last_front = None

    def on_start(self, ctx: object | None = None) -> None:
        return None

    def on_generation(self, generation: int, F=None, X=None, stats: dict[str, Any] | None = None) -> None:
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
            self._records.append({
                "evals": float(self.checkpoints[self._next_idx]),
                "seconds": float(seconds),
                "hypervolume": float(hv),
            })
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
            self._records.append({
                "evals": float(self.checkpoints[self._next_idx]),
                "seconds": float(seconds),
                "hypervolume": float(hv),
            })
            self._next_idx += 1

    def records(self) -> list[dict[str, float]]:
        return list(self._records)

# =============================================================================
# Helpers
# =============================================================================

def _as_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)

def _as_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)

def _as_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None else str(raw)

def _as_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")

def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]

def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def _maybe_dataclass_to_dict(value: object) -> object:
    return asdict(value) if is_dataclass(value) else value


def _load_aos_runtime_options() -> AOSRuntimeOptions:
    """Parse AOS runtime overrides from environment variables."""
    method = _as_str_env("VAMOS_MIC_AOS_METHOD", "epsilon_greedy").strip().lower()
    if method not in VALID_AOS_METHODS:
        raise ValueError(f"Unsupported VAMOS_MIC_AOS_METHOD '{method}'. Supported: {sorted(VALID_AOS_METHODS)}")

    epsilon = _as_float_env("VAMOS_MIC_AOS_EPSILON", 0.05)
    c = _as_float_env("VAMOS_MIC_AOS_C", 1.0)
    gamma = _as_float_env("VAMOS_MIC_AOS_GAMMA", 0.2)
    min_usage = _as_int_env("VAMOS_MIC_AOS_MIN_USAGE", 0)
    floor_prob = _as_float_env("VAMOS_MIC_AOS_FLOOR_PROB", 0.0)
    disable_manyobj = _as_bool_env("VAMOS_MIC_AOS_DISABLE_MANYOBJ", False)

    # Defaults tuned for non-stationary policies; harmless for others.
    default_window = 50 if method in {"sliding_ucb", "thompson_sampling"} else 0
    window_size = _as_int_env("VAMOS_MIC_AOS_WINDOW_SIZE", default_window)

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("VAMOS_MIC_AOS_EPSILON must be in [0, 1].")
    if c < 0.0:
        raise ValueError("VAMOS_MIC_AOS_C must be >= 0.")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("VAMOS_MIC_AOS_GAMMA must be in (0, 1].")
    if min_usage < 0:
        raise ValueError("VAMOS_MIC_AOS_MIN_USAGE must be >= 0.")
    if window_size < 0:
        raise ValueError("VAMOS_MIC_AOS_WINDOW_SIZE must be >= 0.")
    if method == "sliding_ucb" and window_size <= 0:
        raise ValueError("sliding_ucb requires VAMOS_MIC_AOS_WINDOW_SIZE > 0.")
    if not (0.0 <= floor_prob <= 1.0):
        raise ValueError("VAMOS_MIC_AOS_FLOOR_PROB must be in [0, 1].")

    return AOSRuntimeOptions(
        method=method,
        epsilon=epsilon,
        c=c,
        gamma=gamma,
        min_usage=min_usage,
        window_size=window_size,
        floor_prob=floor_prob,
        disable_manyobj=disable_manyobj,
    )


def _load_archive_runtime_options() -> ArchiveRuntimeOptions:
    """Parse archive runtime overrides from environment variables."""
    size = _as_int_env("VAMOS_MIC_ARCHIVE_SIZE", POP_SIZE)
    archive_type = _as_str_env("VAMOS_MIC_ARCHIVE_TYPE", "hypervolume").strip().lower()
    unbounded = _as_bool_env("VAMOS_MIC_ARCHIVE_UNBOUNDED", False)

    if size <= 0:
        raise ValueError("VAMOS_MIC_ARCHIVE_SIZE must be > 0.")
    if archive_type not in {"hypervolume", "crowding"}:
        raise ValueError("VAMOS_MIC_ARCHIVE_TYPE must be one of: hypervolume,crowding.")

    return ArchiveRuntimeOptions(size=size, archive_type=archive_type, unbounded=unbounded)

# =============================================================================
# Reference front generation
# =============================================================================

def generate_reference_fronts() -> None:
    """
    Generate approximate reference fronts for all 21 problems.

    Runs baseline NSGA-II with a large budget and many seeds, then merges
    all non-dominated solutions per problem into a single reference front.
    """
    n_evals = _as_int_env("VAMOS_MIC_REF_N_EVALS", 200_000)
    n_seeds = _as_int_env("VAMOS_MIC_REF_N_SEEDS", 50)
    n_jobs = int(os.environ.get("VAMOS_MIC_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
    engine = _as_str_env("VAMOS_MIC_ENGINE", "numba")

    print(f"Generating reference fronts for {len(ALL_PROBLEMS)} problems")
    print(f"  Budget: {n_evals:,} evals × {n_seeds} seeds")
    print(f"  Engine: {engine}, Workers: {n_jobs}")

    REFERENCE_FRONTS_DIR.mkdir(parents=True, exist_ok=True)

    def _run_one(problem: str, seed: int) -> tuple[str, np.ndarray | None]:
        sel = make_problem_selection(problem)
        prob = sel.instantiate()
        n_var = sel.n_var
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(POP_SIZE)
            .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
            .mutation("pm", prob=1.0 / max(n_var, 1), eta=MUTATION_ETA)
            .selection("tournament")
            .build()
        )
        res = optimize(
            prob,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("max_evaluations", n_evals),
            seed=seed,
            engine=engine,
        )
        return problem, res.F

    tasks = [(p, s) for p in ALL_PROBLEMS for s in range(n_seeds)]
    fronts_by_problem: dict[str, list[np.ndarray]] = defaultdict(list)

    if n_jobs == 1:
        bar = ProgressBar(total=len(tasks), desc="Ref fronts")
        for problem, seed in tasks:
            name, F = _run_one(problem, seed)
            if F is not None:
                fronts_by_problem[name].append(np.asarray(F, dtype=float))
            bar.update(1)
        bar.close()
    else:
        with joblib_progress(total=len(tasks), desc="Ref fronts"):
            results = Parallel(n_jobs=n_jobs, batch_size=1)(
                delayed(_run_one)(p, s) for p, s in tasks
            )
        for name, F in results:
            if F is not None:
                fronts_by_problem[name].append(np.asarray(F, dtype=float))

    # Merge and save
    for problem in ALL_PROBLEMS:
        collected = fronts_by_problem.get(problem, [])
        if not collected:
            print(f"  WARNING: no fronts for {problem}")
            continue
        merged = np.vstack(collected)
        nd = pareto_filter(merged)
        if nd is None or nd.size == 0:
            print(f"  WARNING: empty reference front for {problem}")
            continue
        out = REFERENCE_FRONTS_DIR / f"{problem}.csv"
        np.savetxt(out, nd, delimiter=",")
        print(f"  {problem}: {nd.shape[0]} non-dominated points ({nd.shape[1]} obj) -> {out}")

    print("Reference front generation complete.")

# =============================================================================
# Single run
# =============================================================================

def run_single(
    variant: str,
    problem_name: str,
    seed: int,
    *,
    n_evals: int,
    engine: str,
    checkpoints: list[int] | None,
    capture_aos_trace: bool = False,
    aos_options: AOSRuntimeOptions | None = None,
    archive_options: ArchiveRuntimeOptions | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute one (variant, problem, seed) run."""
    sel = make_problem_selection(problem_name)
    problem = sel.instantiate()
    n_var = sel.n_var

    algo_cfg = build_config(
        variant=variant,
        seed=seed,
        n_var=n_var,
        problem_name=problem_name,
        aos_options=aos_options,
        archive_options=archive_options,
    )
    start = time.perf_counter()
    recorder = (
        HVCheckpointRecorder(problem_name=problem_name, checkpoints=checkpoints, start_time=start)
        if checkpoints
        else None
    )
    res = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("max_evaluations", n_evals),
        seed=seed,
        engine=engine,
        live_viz=recorder,
    )
    elapsed = time.perf_counter() - start
    hv = compute_hv(res.F, problem_name) if res.F is not None else float("nan")

    final_row: dict[str, Any] = {
        "variant": variant,
        "problem": problem_name,
        "n_obj": _N_OBJ.get(problem_name, 0),
        "obj_group": obj_group(problem_name),
        "algorithm": "NSGA-II",
        "engine": engine,
        "n_evals": n_evals,
        "seed": seed,
        "runtime_seconds": float(elapsed),
        "n_solutions": int(res.X.shape[0]) if res.X is not None else 0,
        "hypervolume": float(hv),
        "archive_active": int((getattr(algo_cfg, "archive", None) or {}).get("size", 0) > 0),
        "aos_active": int(getattr(algo_cfg, "adaptive_operator_selection", None) is not None),
        "algorithm_config": json.dumps(_maybe_dataclass_to_dict(algo_cfg), sort_keys=True),
    }

    # Anytime checkpoints
    chk_rows: list[dict[str, Any]] = []
    if recorder is not None:
        for r in recorder.records():
            chk_rows.append({
                "variant": variant,
                "problem": problem_name,
                "obj_group": obj_group(problem_name),
                "engine": engine,
                "n_evals": int(n_evals),
                "seed": int(seed),
                "evals": int(r["evals"]),
                "runtime_seconds": float(r["seconds"]),
                "hypervolume": float(r["hypervolume"]),
            })

    # AOS trace rows
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
                        trace_rows.append({
                            "variant": variant,
                            "problem": problem_name,
                            "obj_group": obj_group(problem_name),
                            "engine": engine,
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
                        })
        except Exception:
            trace_rows = []

    return final_row, chk_rows, trace_rows

# =============================================================================
# Main experiment
# =============================================================================

def run_experiment() -> None:
    """Run configured MIC comparisons on 21 engineering problems."""
    # --- Configuration ---
    n_evals = _as_int_env("VAMOS_MIC_N_EVALS", 50_000)
    n_seeds = _as_int_env("VAMOS_MIC_N_SEEDS", 30)
    engine = _as_str_env("VAMOS_MIC_ENGINE", "numba")
    n_jobs = int(os.environ.get("VAMOS_MIC_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))

    exp_dir = ROOT_DIR / "experiments" / "mic"
    output_csv = Path(_as_str_env("VAMOS_MIC_OUTPUT_CSV", str(exp_dir / "mic_ablation.csv")))

    anytime_csv_raw = os.environ.get("VAMOS_MIC_ANYTIME_CSV")
    if anytime_csv_raw is None:
        anytime_csv_raw = str(exp_dir / "mic_anytime.csv")
    anytime_csv_raw = str(anytime_csv_raw).strip()
    anytime_csv: Path | None = None
    if anytime_csv_raw and anytime_csv_raw not in {"0", "false", "False"}:
        anytime_csv = Path(anytime_csv_raw)

    trace_csv_raw = os.environ.get("VAMOS_MIC_TRACE_CSV")
    if trace_csv_raw is None:
        trace_csv_raw = str(exp_dir / "mic_trace.csv")
    trace_csv_raw = str(trace_csv_raw).strip()
    trace_csv: Path | None = None
    if trace_csv_raw and trace_csv_raw not in {"0", "false", "False"}:
        trace_csv = Path(trace_csv_raw)

    trace_variants = set(
        v.strip().lower()
        for v in _parse_csv_list(_as_str_env("VAMOS_MIC_TRACE_VARIANTS", "aos"))
    )
    trace_problems = set(
        p.strip().lower()
        for p in _parse_csv_list(_as_str_env("VAMOS_MIC_TRACE_PROBLEMS", "re37,rwa2,rwa9"))
    )

    checkpoints: list[int] | None = None
    if anytime_csv is not None:
        raw = _as_str_env("VAMOS_MIC_CHECKPOINTS", "5000,10000,20000,50000")
        checkpoints = sorted(set(int(x) for x in _parse_int_list(raw) if 0 < int(x) <= n_evals))
        if not checkpoints:
            raise ValueError("Checkpoints must contain at least one positive integer.")
        if n_evals not in checkpoints:
            checkpoints.append(int(n_evals))
        checkpoints = sorted(set(checkpoints))

    variants = _parse_csv_list(_as_str_env("VAMOS_MIC_VARIANTS", "baseline,random,aos"))
    variants = [v.strip().lower() for v in variants]
    for v in variants:
        if v not in VALID_VARIANTS:
            raise ValueError(f"Unsupported variant '{v}'. Supported: {sorted(VALID_VARIANTS)}")
    aos_options: AOSRuntimeOptions | None = None
    if any(v == "aos" or v.startswith("aos_") for v in variants):
        aos_options = _load_aos_runtime_options()
    archive_options: ArchiveRuntimeOptions | None = None
    if any(VARIANT_SPECS[v].use_archive for v in variants):
        archive_options = _load_archive_runtime_options()

    problems = list(ALL_PROBLEMS)
    problems_raw = os.environ.get("VAMOS_MIC_PROBLEMS")
    if problems_raw:
        problems = _parse_csv_list(problems_raw)

    # --- Verify reference fronts ---
    missing = [p for p in problems if not (REFERENCE_FRONTS_DIR / f"{p}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing reference fronts for: {missing}. "
            f"Run with --generate-ref-fronts first."
        )

    # --- Print config ---
    print(f"Problems: {len(problems)} ({', '.join(problems)})")
    print(f"Variants: {variants}")
    print(f"Evaluations per run: {n_evals:,}")
    print(f"Seeds: {n_seeds}")
    print(f"Engine: {engine}")
    print(f"Parallel workers: {n_jobs}")
    if aos_options is not None:
        print(
            "AOS runtime policy: "
            f"method={aos_options.method}, epsilon={aos_options.epsilon}, c={aos_options.c}, "
            f"gamma={aos_options.gamma}, min_usage={aos_options.min_usage}, "
            f"window_size={aos_options.window_size}, floor_prob={aos_options.floor_prob}, "
            f"disable_manyobj={int(aos_options.disable_manyobj)}"
        )
    if archive_options is not None:
        print(
            "Archive runtime options: "
            f"size={archive_options.size}, type={archive_options.archive_type}, "
            f"unbounded={int(archive_options.unbounded)}"
        )
    if anytime_csv:
        print(f"Anytime checkpoints: {checkpoints}")
    if trace_csv:
        print(f"AOS trace problems: {sorted(trace_problems)}")

    # --- Resume support ---
    resume = int(os.environ.get("VAMOS_MIC_RESUME", "0")) != 0
    tasks = [(v, p, s) for v in variants for p in problems for s in range(n_seeds)]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if anytime_csv:
        anytime_csv.parent.mkdir(parents=True, exist_ok=True)
    if trace_csv:
        trace_csv.parent.mkdir(parents=True, exist_ok=True)

    if not resume:
        for f in [output_csv, anytime_csv, trace_csv]:
            if f is not None and f.exists():
                f.unlink()

    written_final = 0
    written_anytime = 0
    written_trace = 0

    done: set[tuple[str, str, int]] = set()
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        written_final = len(existing)
        if {"variant", "problem", "seed"}.issubset(existing.columns):
            if "n_evals" in existing.columns:
                existing = existing[existing["n_evals"].astype(int) == n_evals]
            existing["variant"] = existing["variant"].astype(str).str.strip().str.lower()
            existing["problem"] = existing["problem"].astype(str).str.strip().str.lower()
            existing["seed"] = existing["seed"].astype(int)
            done = set(zip(existing["variant"], existing["problem"], existing["seed"]))
            if done:
                print(f"Resume: skipping {len(done)} completed runs")

    if resume and anytime_csv and anytime_csv.exists():
        try:
            written_anytime = len(pd.read_csv(anytime_csv))
        except Exception:
            written_anytime = 0

    if resume and trace_csv and trace_csv.exists():
        try:
            written_trace = len(pd.read_csv(trace_csv))
        except Exception:
            written_trace = 0

    if done:
        tasks = [t for t in tasks if t not in done]

    print(f"Total runs: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        return

    # --- Time-based checkpointing ---
    checkpoint_interval = float(_as_int_env("VAMOS_CHECKPOINT_INTERVAL_MIN", 30)) * 60.0
    last_ckpt_time = time.perf_counter()
    pending_final: list[dict[str, Any]] = []
    pending_anytime: list[dict[str, Any]] = []
    pending_trace: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal written_final, written_anytime, written_trace
        nonlocal pending_final, pending_anytime, pending_trace, last_ckpt_time
        if pending_final:
            pd.DataFrame(pending_final).to_csv(output_csv, mode="a", header=(written_final == 0), index=False)
            written_final += len(pending_final)
            pending_final = []
        if anytime_csv and pending_anytime:
            pd.DataFrame(pending_anytime).to_csv(anytime_csv, mode="a", header=(written_anytime == 0), index=False)
            written_anytime += len(pending_anytime)
            pending_anytime = []
        if trace_csv and pending_trace:
            pd.DataFrame(pending_trace).to_csv(trace_csv, mode="a", header=(written_trace == 0), index=False)
            written_trace += len(pending_trace)
            pending_trace = []
        last_ckpt_time = time.perf_counter()

    def _append(row: dict[str, Any], chk: list[dict[str, Any]], tr: list[dict[str, Any]], *, force: bool = False) -> None:
        nonlocal last_ckpt_time
        pending_final.append(row)
        pending_anytime.extend(chk)
        pending_trace.extend(tr)
        if force or (time.perf_counter() - last_ckpt_time) >= checkpoint_interval:
            _flush()

    # --- Execute ---
    if n_jobs == 1:
        bar = ProgressBar(total=len(tasks), desc="MIC experiment")
        for i, (variant, problem, seed) in enumerate(tasks):
            capture = trace_csv is not None and variant in trace_variants and problem in trace_problems
            row, chk, tr = run_single(
                variant, problem, seed,
                n_evals=n_evals, engine=engine,
                checkpoints=checkpoints, capture_aos_trace=capture,
                aos_options=aos_options,
                archive_options=archive_options,
            )
            is_last = (i == len(tasks) - 1)
            next_seed = tasks[i + 1][2] if not is_last else None
            force = is_last or (next_seed != seed)
            _append(row, chk, tr, force=force)
            bar.update(1)
        bar.close()
    else:
        with joblib_progress(total=len(tasks), desc="MIC experiment"):
            parallel = Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator")
            for row, chk, tr in parallel(
                delayed(run_single)(
                    v, p, s,
                    n_evals=n_evals, engine=engine,
                    checkpoints=checkpoints,
                    capture_aos_trace=(trace_csv is not None and v in trace_variants and p in trace_problems),
                    aos_options=aos_options,
                    archive_options=archive_options,
                )
                for v, p, s in tasks
            ):
                _append(row, chk, tr)
        _flush()

    print(f"Wrote {written_final} rows -> {output_csv}")
    if anytime_csv:
        print(f"Wrote {written_anytime} rows -> {anytime_csv}")
    if trace_csv:
        print(f"Wrote {written_trace} rows -> {trace_csv}")

# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="MIC paper experiment: AOS on RE + RWA problems")
    ap.add_argument("--generate-ref-fronts", action="store_true", help="Generate reference fronts (run first)")
    args = ap.parse_args()

    if args.generate_ref_fronts:
        generate_reference_fronts()
    else:
        run_experiment()


if __name__ == "__main__":
    main()

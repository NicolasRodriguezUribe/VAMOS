"""
Mini sweep for AOS hyperparameters on a small problem subset.

Why
---
Running the full ablation suite (all problems × many seeds × multiple variants) is expensive.
This script runs a small grid over a small subset (default: zdt4, dtlz3, wfg9) to quickly
identify which AOS settings move the needle (and which cause regressions).

Defaults match the paper setup where possible:
  - engine: numba
  - n_evals: 20k
  - checkpoints: 5k,10k,20k
  - operator pools:
      * aos: a small diverse default portfolio
      * tuned_aos: tuned pipeline + (unique) pipelines from top-k tuner candidates

Outputs
-------
  - experiments/aos_sweep_runs.csv: per-run long format
  - experiments/aos_sweep_summary.csv: aggregated summary per (variant, method, epsilon, hv_weight)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning import config_from_assignment
from vamos.foundation.problem.registry import make_problem_selection

try:
    from .benchmark_utils import compute_hv, _reference_hv, _reference_point
    from .progress_utils import ProgressBar
except ImportError:  # pragma: no cover - allow running as a standalone script
    from benchmark_utils import compute_hv, _reference_hv, _reference_point
    from progress_utils import ProgressBar


ROOT_DIR = Path(__file__).resolve().parents[1]

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


def _parse_csv_list(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in _parse_csv_list(raw):
        out.append(float(item))
    return out


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(raw):
        out.append(int(item))
    return out


def problem_dims(problem_name: str) -> tuple[int, int]:
    problem_name = str(problem_name).strip().lower()
    if problem_name in ZDT_N_VAR:
        return int(ZDT_N_VAR[problem_name]), ZDT_N_OBJ
    if problem_name in DTLZ_N_VAR:
        return int(DTLZ_N_VAR[problem_name]), DTLZ_N_OBJ
    if problem_name.startswith("wfg"):
        return WFG_N_VAR, WFG_N_OBJ
    raise ValueError(f"Unknown problem dimensions for '{problem_name}'")


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


def _auc_norm(evals: list[int], hv: list[float], *, max_evals: int) -> float:
    x = np.asarray(evals, dtype=float)
    y = np.asarray(hv, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size <= 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
    area = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
    return area / float(max_evals) if max_evals > 0 else float("nan")


def _baseline_operator_pool(*, n_var: int) -> list[dict[str, Any]]:
    return [
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


def _make_aos_cfg(
    *,
    seed: int,
    n_var: int,
    problem_name: str,
    operator_pool: list[dict[str, Any]],
    method: str,
    epsilon: float,
    ucb_c: float,
    w_survival: float,
    w_nd: float,
    w_hv_delta: float,
) -> dict[str, Any]:
    hv_reference_point: list[float] | None = None
    hv_reference_hv: float | None = None
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
        "method": str(method).strip().lower(),
        "epsilon": float(epsilon),
        "c": float(ucb_c),
        "window_size": 50,
        "min_usage": 0,
        "rng_seed": int(seed),
        "reward_scope": "combined",
        "reward_weights": {
            "survival": float(w_survival),
            "nd_insertions": float(w_nd),
            "hv_delta": float(w_hv_delta),
        },
        "hv_reference_point": hv_reference_point,
        "hv_reference_hv": hv_reference_hv,
        "operator_pool": operator_pool,
    }


def _tuned_operator_pool(
    *,
    tuned: NSGAIIConfig,
    n_var: int,
    topk_path: Path,
    max_arms: int,
) -> list[dict[str, Any]]:
    def _sig(entry: dict[str, Any]) -> str:
        return json.dumps(entry, sort_keys=True)

    pool: list[dict[str, Any]] = [{"crossover": tuned.crossover, "mutation": tuned.mutation}]
    seen = {_sig(pool[0])}

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
                    pool.append(entry)
                    seen.add(sig)
                    if len(pool) >= max_arms:
                        break
        except Exception:
            pass

    if len(pool) < 2:
        pool.append({"crossover": tuned.crossover, "mutation": ("pm", {"prob": 1.0 / n_var, "eta": MUTATION_ETA})})

    return pool


def _build_config(
    *,
    variant: str,
    seed: int,
    n_var: int,
    problem_name: str,
    tuned_cfg: dict[str, Any] | None,
    topk_path: Path,
    max_arms: int,
    method: str,
    epsilon: float,
    ucb_c: float,
    w_survival: float,
    w_nd: float,
    w_hv_delta: float,
) -> NSGAIIConfig:
    variant = str(variant).strip().lower()
    base = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
    )
    if variant == "baseline":
        return base.build()

    if variant == "tuned":
        if tuned_cfg is None:
            raise ValueError("tuned variant requested but tuned_cfg is None")
        return config_from_assignment("nsgaii", tuned_cfg)

    if variant == "aos":
        aos_cfg = _make_aos_cfg(
            seed=seed,
            n_var=n_var,
            problem_name=problem_name,
            operator_pool=_baseline_operator_pool(n_var=n_var),
            method=method,
            epsilon=epsilon,
            ucb_c=ucb_c,
            w_survival=w_survival,
            w_nd=w_nd,
            w_hv_delta=w_hv_delta,
        )
        return base.adaptive_operator_selection(aos_cfg).build()

    if variant == "tuned_aos":
        if tuned_cfg is None:
            raise ValueError("tuned_aos variant requested but tuned_cfg is None")
        tuned = config_from_assignment("nsgaii", tuned_cfg)
        pool = _tuned_operator_pool(tuned=tuned, n_var=n_var, topk_path=topk_path, max_arms=max_arms)
        aos_cfg = _make_aos_cfg(
            seed=seed,
            n_var=n_var,
            problem_name=problem_name,
            operator_pool=pool,
            method=method,
            epsilon=epsilon,
            ucb_c=ucb_c,
            w_survival=w_survival,
            w_nd=w_nd,
            w_hv_delta=w_hv_delta,
        )
        return replace(tuned, adaptive_operator_selection=aos_cfg)

    raise ValueError(f"Unsupported variant '{variant}'.")


def _run_single(
    *,
    variant: str,
    problem_name: str,
    seed: int,
    n_evals: int,
    engine: str,
    tuned_cfg: dict[str, Any] | None,
    topk_path: Path,
    max_arms: int,
    method: str,
    epsilon: float,
    ucb_c: float,
    w_survival: float,
    w_nd: float,
    w_hv_delta: float,
    checkpoints: list[int],
) -> dict[str, Any]:
    n_var, n_obj = problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
    algo_cfg = _build_config(
        variant=variant,
        seed=seed,
        n_var=n_var,
        problem_name=problem_name,
        tuned_cfg=tuned_cfg,
        topk_path=topk_path,
        max_arms=max_arms,
        method=method,
        epsilon=epsilon,
        ucb_c=ucb_c,
        w_survival=w_survival,
        w_nd=w_nd,
        w_hv_delta=w_hv_delta,
    )

    start = time.perf_counter()
    recorder = HVCheckpointRecorder(problem_name=problem_name, checkpoints=checkpoints, start_time=start)
    res = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", n_evals),
        seed=seed,
        engine=engine,
        live_viz=recorder,
    )
    elapsed = float(time.perf_counter() - start)
    hv_final = float(compute_hv(res.F, problem_name)) if getattr(res, "F", None) is not None else float("nan")

    records = recorder.records()
    hv_by_chk = {int(r["evals"]): float(r["hypervolume"]) for r in records if "evals" in r and "hypervolume" in r}
    for c in checkpoints:
        if int(c) not in hv_by_chk:
            hv_by_chk[int(c)] = float("nan")

    auc = _auc_norm([int(c) for c in checkpoints], [float(hv_by_chk[int(c)]) for c in checkpoints], max_evals=int(n_evals))

    row: dict[str, Any] = {
        "variant": str(variant).strip().lower(),
        "problem": str(problem_name).strip().lower(),
        "seed": int(seed),
        "engine": str(engine).strip().lower(),
        "n_evals": int(n_evals),
        "runtime_seconds": float(elapsed),
        "hv_final": float(hv_final),
        "auc": float(auc),
        "method": str(method).strip().lower(),
        "epsilon": float(epsilon),
        "ucb_c": float(ucb_c),
        "w_survival": float(w_survival),
        "w_nd": float(w_nd),
        "w_hv_delta": float(w_hv_delta),
        "max_arms": int(max_arms),
    }
    for c in checkpoints:
        row[f"hv_{int(c)}"] = float(hv_by_chk[int(c)])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=str, default="zdt4,dtlz3,wfg9", help="Comma-separated problems for the sweep.")
    ap.add_argument(
        "--variants",
        type=str,
        default="aos,tuned_aos",
        help="Comma-separated AOS variants to sweep (default: aos,tuned_aos).",
    )
    ap.add_argument("--n-evals", type=int, default=20000, help="Evaluation budget per run (default: 20000).")
    ap.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3).")
    ap.add_argument("--engine", type=str, default="numba", help="Engine (default: numba).")
    ap.add_argument("--checkpoints", type=str, default="5000,10000,20000", help="Comma-separated HV checkpoints.")
    ap.add_argument("--methods", type=str, default="epsilon_greedy,ucb", help="Comma-separated AOS methods to test.")
    ap.add_argument("--epsilons", type=str, default="0.02,0.05,0.1", help="Comma-separated epsilon values to test.")
    ap.add_argument("--hv-weights", type=str, default="0.2,0.4", help="Comma-separated hv_delta weights to test.")
    ap.add_argument("--ucb-c", type=float, default=1.0, help="UCB c parameter (default: 1.0).")
    ap.add_argument("--w-survival", type=float, default=0.40, help="Reward weight for survival (default: 0.40).")
    ap.add_argument("--w-nd", type=float, default=0.40, help="Reward weight for ND insertions (default: 0.40).")
    ap.add_argument(
        "--tuned-json",
        type=str,
        default=str(ROOT_DIR / "experiments" / "tuned_nsgaii.json"),
        help="Path to tuned config JSON (default: experiments/tuned_nsgaii.json).",
    )
    ap.add_argument(
        "--topk-json",
        type=str,
        default=str(ROOT_DIR / "experiments" / "tuned_nsgaii_topk.json"),
        help="Path to top-k tuned configs JSON (default: experiments/tuned_nsgaii_topk.json).",
    )
    ap.add_argument("--max-arms", type=int, default=5, help="Max arms for tuned_aos operator pool (default: 5).")
    ap.add_argument(
        "--include-reference",
        action="store_true",
        default=True,
        help="Also run baseline/tuned once as references (default: enabled).",
    )
    ap.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable baseline/tuned reference runs.",
    )
    ap.add_argument("--out-runs", type=str, default="experiments/aos_sweep_runs.csv", help="Output per-run CSV.")
    ap.add_argument("--out-summary", type=str, default="experiments/aos_sweep_summary.csv", help="Output summary CSV.")
    args = ap.parse_args()

    problems = [p.strip().lower() for p in _parse_csv_list(args.problems)]
    variants = [v.strip().lower() for v in _parse_csv_list(args.variants)]
    checkpoints = sorted(set(int(x) for x in _parse_int_list(args.checkpoints) if int(x) > 0))
    n_evals = int(args.n_evals)
    n_seeds = int(args.seeds)
    engine = str(args.engine).strip().lower()

    methods = [m.strip().lower() for m in _parse_csv_list(args.methods)]
    epsilons = [float(x) for x in _parse_float_list(args.epsilons)]
    hv_weights = [float(x) for x in _parse_float_list(args.hv_weights)]

    include_ref = bool(args.include_reference) and not bool(args.no_reference)

    tuned_path = Path(args.tuned_json)
    tuned_cfg = json.loads(tuned_path.read_text(encoding="utf-8")) if tuned_path.exists() else None
    if tuned_cfg is None:
        raise FileNotFoundError(f"Missing tuned config JSON: {tuned_path}")

    topk_path = Path(args.topk_json)

    tasks: list[dict[str, Any]] = []
    if include_ref:
        for variant in ("baseline", "tuned"):
            for problem in problems:
                for seed in range(n_seeds):
                    tasks.append(
                        {
                            "variant": variant,
                            "problem": problem,
                            "seed": seed,
                            "method": "ref",
                            "epsilon": float("nan"),
                            "w_hv_delta": float("nan"),
                        }
                    )

    for variant in variants:
        for method in methods:
            for eps in epsilons:
                for w_hv in hv_weights:
                    for problem in problems:
                        for seed in range(n_seeds):
                            tasks.append(
                                {
                                    "variant": variant,
                                    "problem": problem,
                                    "seed": seed,
                                    "method": method,
                                    "epsilon": float(eps),
                                    "w_hv_delta": float(w_hv),
                                }
                            )

    print(f"Problems: {problems}")
    print(f"Variants: {variants}" + (" (+ baseline/tuned refs)" if include_ref else ""))
    print(f"Seeds: {n_seeds}")
    print(f"n_evals: {n_evals}")
    print(f"Engine: {engine}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Methods: {methods}")
    print(f"Epsilons: {epsilons}")
    print(f"HV weights: {hv_weights}")
    print(f"Total runs: {len(tasks)}")

    bar = ProgressBar(total=len(tasks), desc="AOS sweep runs")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        variant = str(task["variant"])
        problem = str(task["problem"])
        seed = int(task["seed"])
        method = str(task["method"])
        eps = float(task["epsilon"])
        w_hv = float(task["w_hv_delta"])

        # Reference runs ignore AOS settings (but keep columns present for analysis).
        method_eff = "epsilon_greedy" if method == "ref" else method
        eps_eff = 0.0 if method == "ref" else eps
        w_hv_eff = 0.0 if method == "ref" else w_hv

        try:
            row = _run_single(
                variant=variant,
                problem_name=problem,
                seed=seed,
                n_evals=n_evals,
                engine=engine,
                tuned_cfg=tuned_cfg,
                topk_path=topk_path,
                max_arms=int(args.max_arms),
                method=method_eff,
                epsilon=eps_eff,
                ucb_c=float(args.ucb_c),
                w_survival=float(args.w_survival),
                w_nd=float(args.w_nd),
                w_hv_delta=w_hv_eff,
                checkpoints=checkpoints,
            )
            row["is_reference"] = int(1 if method == "ref" else 0)
            # Preserve the "grid keys" for grouping (including ref rows).
            row["method"] = method
            row["epsilon"] = eps
            row["w_hv_delta"] = w_hv
            rows.append(row)
        except Exception as exc:
            fail = {
                "variant": str(variant).strip().lower(),
                "problem": str(problem).strip().lower(),
                "seed": int(seed),
                "engine": str(engine).strip().lower(),
                "n_evals": int(n_evals),
                "runtime_seconds": float("nan"),
                "hv_final": float("nan"),
                "auc": float("nan"),
                "method": method,
                "epsilon": eps,
                "ucb_c": float(args.ucb_c),
                "w_survival": float(args.w_survival),
                "w_nd": float(args.w_nd),
                "w_hv_delta": w_hv,
                "max_arms": int(args.max_arms),
                "is_reference": int(1 if method == "ref" else 0),
                "error": repr(exc),
            }
            for c in checkpoints:
                fail[f"hv_{int(c)}"] = float("nan")
            rows.append(fail)
        bar.update(1)
    bar.close()

    runs_df = pd.DataFrame(rows)
    out_runs = Path(args.out_runs)
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(out_runs, index=False)
    print(f"Wrote runs: {out_runs}")

    # Summary: compare AOS variants against their natural references:
    # - aos vs baseline
    # - tuned_aos vs tuned
    summary_rows: list[dict[str, Any]] = []

    # References by (problem,seed)
    baseline_ref = runs_df[runs_df["variant"] == "baseline"][["problem", "seed", "hv_final", "auc", *[f"hv_{c}" for c in checkpoints]]].copy()
    tuned_ref = runs_df[runs_df["variant"] == "tuned"][["problem", "seed", "hv_final", "auc", *[f"hv_{c}" for c in checkpoints]]].copy()
    baseline_ref = baseline_ref.rename(columns={c: f"ref_{c}" for c in baseline_ref.columns if c not in {"problem", "seed"}})
    tuned_ref = tuned_ref.rename(columns={c: f"ref_{c}" for c in tuned_ref.columns if c not in {"problem", "seed"}})

    def _summarize_variant(v: str, ref: pd.DataFrame | None) -> None:
        df_v = runs_df[runs_df["variant"] == v].copy()
        df_v = df_v[df_v.get("is_reference", 0) == 0].copy()
        if df_v.empty:
            return

        if ref is not None and not ref.empty:
            df_v = df_v.merge(ref, on=["problem", "seed"], how="left")
            df_v["delta_hv_final"] = df_v["hv_final"] - df_v["ref_hv_final"]
            df_v["delta_auc"] = df_v["auc"] - df_v["ref_auc"]
            for c in checkpoints:
                df_v[f"delta_hv_{c}"] = df_v[f"hv_{c}"] - df_v[f"ref_hv_{c}"]
        else:
            df_v["delta_hv_final"] = float("nan")
            df_v["delta_auc"] = float("nan")
            for c in checkpoints:
                df_v[f"delta_hv_{c}"] = float("nan")

        group_keys = ["variant", "method", "epsilon", "w_hv_delta"]

        # First aggregate across seeds per problem (median), then average across problems.
        per_problem = df_v.groupby([*group_keys, "problem"], as_index=False).agg(
            hv_final=("hv_final", "median"),
            auc=("auc", "median"),
            delta_hv_final=("delta_hv_final", "median"),
            delta_auc=("delta_auc", "median"),
            **{f"hv_{c}": (f"hv_{c}", "median") for c in checkpoints},
            **{f"delta_hv_{c}": (f"delta_hv_{c}", "median") for c in checkpoints},
        )

        agg = per_problem.groupby(group_keys).agg(
            avg_hv_final=("hv_final", "mean"),
            min_hv_final=("hv_final", "min"),
            avg_auc=("auc", "mean"),
            min_auc=("auc", "min"),
            avg_delta_hv_final=("delta_hv_final", "mean"),
            min_delta_hv_final=("delta_hv_final", "min"),
            avg_delta_auc=("delta_auc", "mean"),
            min_delta_auc=("delta_auc", "min"),
            **{f"avg_hv_{c}": (f"hv_{c}", "mean") for c in checkpoints},
            **{f"min_hv_{c}": (f"hv_{c}", "min") for c in checkpoints},
            **{f"avg_delta_hv_{c}": (f"delta_hv_{c}", "mean") for c in checkpoints},
            **{f"min_delta_hv_{c}": (f"delta_hv_{c}", "min") for c in checkpoints},
        )
        agg = agg.reset_index()

        # Add a few explicit per-problem columns for debugging (e.g., dtlz3 collapse).
        for probe in ["dtlz3", "zdt4", "wfg9"]:
            sub = per_problem[per_problem["problem"] == probe][[*group_keys, "hv_final", "auc"]].copy()
            if sub.empty:
                continue
            sub = sub.rename(columns={"hv_final": f"{probe}_hv_final", "auc": f"{probe}_auc"})
            agg = agg.merge(sub, on=group_keys, how="left")

        for _, r in agg.iterrows():
            summary_rows.append({k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in agg.columns})

    if include_ref:
        _summarize_variant("aos", baseline_ref)
        _summarize_variant("tuned_aos", tuned_ref)
    else:
        _summarize_variant("aos", None)
        _summarize_variant("tuned_aos", None)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["variant", "avg_delta_auc", "avg_delta_hv_final"], ascending=[True, False, False])

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary, index=False)
    print(f"Wrote summary: {out_summary}")

    if not summary_df.empty:
        print("\nTop-5 per variant (by avg_delta_auc, then avg_delta_hv_final):")
        for v in sorted(summary_df["variant"].unique()):
            top = summary_df[summary_df["variant"] == v].head(5).copy()
            keep = [
                "variant",
                "method",
                "epsilon",
                "w_hv_delta",
                "avg_delta_auc",
                "avg_delta_hv_final",
                "min_delta_hv_final",
                "dtlz3_hv_final",
                "zdt4_hv_final",
                "wfg9_hv_final",
            ]
            keep = [c for c in keep if c in top.columns]
            print(f"\n[{v}]")
            with pd.option_context("display.max_columns", None, "display.width", 140):
                print(top[keep].to_string(index=False, float_format=lambda x: f"{x:.4f}" if np.isfinite(x) else "nan"))


if __name__ == "__main__":
    main()

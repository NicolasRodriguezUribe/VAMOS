from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vamos import optimize
from vamos.engine.adaptation.online_control import DEFAULT_PROTOTYPE_SET, OperatorFamily, Regime, available_intent_prototypes
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.moocore_indicators import get_indicator, has_moocore
from vamos.resources import weight_path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "online_control_pilot.json"
FAMILY_LABELS = tuple(family.value for family in OperatorFamily)
REGIME_LABELS = tuple(regime.value for regime in Regime)
INTENT_LABELS = available_intent_prototypes(DEFAULT_PROTOTYPE_SET)
BASELINE_BY_HOST = {"nsgaii": "fixed_sbx", "moead": "fixed_de"}


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value if value is None or isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)
                    for key, value in row.items()
                }
            )


def _variant_group(variant: str) -> str:
    if variant.startswith("fixed_"):
        return "fixed"
    if variant.startswith("adaptive_") or variant == "semantic_prototype_sbx":
        return "adaptive"
    return "semantic_static"


def _build_online_control_payload(variant: str, credit_model: str) -> dict[str, Any] | None:
    if variant == "fixed_sbx" or variant == "fixed_de":
        return None
    if variant == "semantic_prototype_sbx":
        return {
            "enabled": True,
            "policy": "adaptive_flat_parameter",
            "credit_model": credit_model,
            "trace_level": "basic",
            "fixed_family": "sbx_like",
        }
    if variant == "flat_operator":
        return {"enabled": True, "policy": "flat_operator", "credit_model": credit_model, "trace_level": "basic"}
    if variant == "adaptive_flat_operator":
        return {"enabled": True, "policy": "adaptive_flat_operator", "credit_model": credit_model, "trace_level": "basic"}
    if variant == "flat_parameter":
        return {
            "enabled": True,
            "policy": "flat_parameter",
            "credit_model": credit_model,
            "trace_level": "basic",
            "fixed_family": "sbx_like",
        }
    if variant == "adaptive_flat_parameter":
        return {
            "enabled": True,
            "policy": "adaptive_flat_parameter",
            "credit_model": credit_model,
            "trace_level": "basic",
            "fixed_family": "sbx_like",
        }
    if variant == "hierarchical_joint":
        return {"enabled": True, "policy": "hierarchical_joint", "credit_model": credit_model, "trace_level": "basic"}
    if variant == "adaptive_hierarchical_joint":
        return {"enabled": True, "policy": "adaptive_hierarchical_joint", "credit_model": credit_model, "trace_level": "basic"}
    if variant == "adaptive_hierarchical_joint_no_regime":
        return {
            "enabled": True,
            "router": "static_expand",
            "policy": "adaptive_hierarchical_joint",
            "credit_model": credit_model,
            "trace_level": "basic",
        }
    if variant == "adaptive_hierarchical_joint_fixed_family_sbx":
        return {
            "enabled": True,
            "policy": "adaptive_hierarchical_joint",
            "credit_model": credit_model,
            "trace_level": "basic",
            "fixed_family": "sbx_like",
        }
    if variant == "adaptive_hierarchical_joint_fixed_family_de":
        return {
            "enabled": True,
            "policy": "adaptive_hierarchical_joint",
            "credit_model": credit_model,
            "trace_level": "basic",
            "fixed_family": "de_like",
        }
    raise ValueError(f"Unsupported pilot variant '{variant}'.")


def _normalize_problem_specs(raw_problems: list[Any], default_n_var: int) -> list[dict[str, Any]]:
    problem_specs: list[dict[str, Any]] = []
    for item in raw_problems:
        if isinstance(item, str):
            problem_specs.append({"key": str(item), "n_var": default_n_var})
            continue
        if isinstance(item, dict):
            key = str(item.get("key", "")).strip()
            if not key:
                raise ValueError("Problem spec mappings must include a non-empty 'key'.")
            spec = {"key": key, "n_var": int(item.get("n_var", default_n_var))}
            if item.get("n_obj") is not None:
                spec["n_obj"] = int(item["n_obj"])
            if item.get("suite") is not None:
                spec["suite"] = str(item["suite"]).strip().lower()
            problem_specs.append(spec)
            continue
        raise TypeError("Problem entries must be either strings or mappings.")
    return problem_specs


def _build_nsgaii_config(variant: str, *, pop_size: int, n_var: int, credit_model: str) -> NSGAIIConfig:
    mut_prob = 1.0 / float(n_var)
    crossover = ("de", {"CR": 1.0, "F": 0.5}) if variant == "fixed_de" else ("sbx", {"prob": 1.0, "eta": 20.0})
    builder = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover(crossover[0], **crossover[1])
        .mutation("polynomial", prob=mut_prob, eta=20.0)
        .selection("tournament", size=2)
        .result_mode("non_dominated")
    )
    online_control = _build_online_control_payload(variant, credit_model)
    if online_control is not None:
        builder.online_control(online_control)
    return builder.build()


def _build_moead_config(variant: str, *, pop_size: int, n_var: int, n_obj: int, credit_model: str) -> MOEADConfig:
    mut_prob = 1.0 / float(n_var)
    crossover = ("de", {"cr": 1.0, "f": 0.5}) if variant == "fixed_de" else ("sbx", {"prob": 1.0, "eta": 20.0})
    builder = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .batch_size(1)
        .neighbor_size(min(10, pop_size))
        .delta(0.9)
        .replace_limit(2)
        .crossover(crossover[0], **crossover[1])
        .mutation("polynomial", prob=mut_prob, eta=20.0)
        .aggregation("pbi", theta=5.0)
        .result_mode("non_dominated")
    )
    if n_obj > 2:
        builder.weight_vectors(path=str(weight_path("W3D_91.dat").parent))
    online_control = _build_online_control_payload(variant, credit_model)
    if online_control is not None:
        builder.online_control(online_control)
    return builder.build()


def _build_algorithm_config(host: str, variant: str, *, pop_size: int, n_var: int, n_obj: int, credit_model: str) -> Any:
    if host == "nsgaii":
        return _build_nsgaii_config(variant, pop_size=pop_size, n_var=n_var, credit_model=credit_model)
    if host == "moead":
        return _build_moead_config(variant, pop_size=pop_size, n_var=n_var, n_obj=n_obj, credit_model=credit_model)
    raise ValueError(f"Unsupported host '{host}'.")


def _load_reference_front(problem_key: str, n_obj: int) -> np.ndarray | None:
    front_path = resolve_reference_front_path(problem_key, None, n_obj=n_obj)
    if front_path is None:
        return None
    front = np.loadtxt(front_path, delimiter=",")
    front = np.atleast_2d(np.asarray(front, dtype=float))
    if front.ndim != 2 or front.shape[0] == 0:
        return None
    return front


def _flatten_share_columns(row: dict[str, Any], prefix: str, labels: tuple[str, ...], shares: dict[str, float] | None) -> None:
    share_map = shares or {}
    for label in labels:
        row[f"{prefix}_{label}"] = float(share_map.get(label, 0.0))


def _baseline_family_shares(variant: str) -> dict[str, float]:
    if variant == "fixed_sbx":
        return {"sbx_like": 1.0}
    if variant == "fixed_de":
        return {"de_like": 1.0}
    return {}


def _flatten_runtime_profile(row: dict[str, Any], runtime_profile: dict[str, Any] | None) -> None:
    profile = runtime_profile or {}
    expected = (
        "controller_time_ms",
        "start_step_time_ms",
        "router_time_ms",
        "policy_select_time_ms",
        "policy_update_time_ms",
        "trace_time_ms",
        "decode_time_ms",
        "variation_time_ms",
        "evaluation_time_ms",
        "survival_time_ms",
        "total_runtime_ms",
    )
    for key in expected:
        value = profile.get(key)
        row[f"profile_{key}"] = float(value) if isinstance(value, (int, float)) else None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _run_variant(
    *,
    host: str,
    variant: str,
    problem_key: str,
    n_var: int,
    n_obj: int | None,
    suite: str | None,
    pop_size: int,
    max_evaluations: int,
    seed: int,
    engine: str,
    credit_model: str,
) -> dict[str, Any]:
    selection = make_problem_selection(problem_key, n_var=n_var, n_obj=n_obj)
    problem = selection.instantiate()
    cfg = _build_algorithm_config(
        host,
        variant,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=selection.n_obj,
        credit_model=credit_model,
    )
    t0 = time.perf_counter()
    result = optimize(
        problem,
        algorithm=host,
        algorithm_config=cfg,
        termination=("max_evaluations", max_evaluations),
        seed=seed,
        engine=engine,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    front = np.asarray(result.F, dtype=float) if result.F is not None else np.empty((0, selection.n_obj), dtype=float)
    payload = result.data
    online = payload.get("online_control") if isinstance(payload, dict) else None
    run_summary = online.get("run_summary", {}) if isinstance(online, dict) else {}
    trace_rows = online.get("trace_rows", []) if isinstance(online, dict) else []
    policy_state = online.get("policy_state") if isinstance(online, dict) else None
    runtime_profile = run_summary.get("runtime_profile") if isinstance(run_summary.get("runtime_profile"), dict) else {}
    controller_profile = online.get("controller_profile") if isinstance(online, dict) and isinstance(online.get("controller_profile"), dict) else {}
    run_id = f"{host}__{problem_key}__{variant}__seed{seed}"
    row: dict[str, Any] = {
        "run_id": run_id,
        "host": host,
        "variant": variant,
        "variant_group": _variant_group(variant),
        "problem": problem_key,
        "suite": suite,
        "problem_n_var": selection.n_var,
        "problem_n_obj": selection.n_obj,
        "seed": seed,
        "engine": engine,
        "population_size": pop_size,
        "max_evaluations": max_evaluations,
        "evaluations": int(payload.get("evaluations", max_evaluations)),
        "time_ms": elapsed_ms,
        "n_solutions": int(front.shape[0]),
        "online_control_enabled": isinstance(online, dict),
        "policy": run_summary.get("policy"),
        "credit_model": run_summary.get("credit_model") or (credit_model if not variant.startswith("fixed_") else None),
        "average_reward": run_summary.get("average_reward"),
        "average_bounded_reward": run_summary.get("average_bounded_reward"),
        "average_overhead_ms": run_summary.get("average_overhead_ms"),
        "family_switches": run_summary.get("family_switches", 0),
        "intent_switches": run_summary.get("intent_switches", 0),
        "regime_switches": run_summary.get("regime_switches", 0),
        "family_concentration": run_summary.get("family_concentration", 1.0 if variant.startswith("fixed_") else None),
        "intent_concentration": run_summary.get("intent_concentration"),
        "regime_concentration": run_summary.get("regime_concentration"),
        "trace_steps": len(trace_rows),
        "policy_state_available": isinstance(policy_state, dict),
        "_front": front,
        "_trace_rows": trace_rows,
        "_policy_state": policy_state if isinstance(policy_state, dict) else None,
    }
    combined_runtime_profile = dict(controller_profile) if isinstance(controller_profile, dict) else {}
    if isinstance(runtime_profile, dict):
        combined_runtime_profile.update(runtime_profile)
    _flatten_runtime_profile(row, combined_runtime_profile)
    family_shares = run_summary.get("family_shares") if isinstance(run_summary.get("family_shares"), dict) else _baseline_family_shares(variant)
    _flatten_share_columns(row, "family_share", FAMILY_LABELS, family_shares)
    _flatten_share_columns(
        row,
        "regime_share",
        REGIME_LABELS,
        run_summary.get("regime_shares") if isinstance(run_summary.get("regime_shares"), dict) else {},
    )
    _flatten_share_columns(
        row,
        "intent_share",
        INTENT_LABELS,
        run_summary.get("intent_shares") if isinstance(run_summary.get("intent_shares"), dict) else {},
    )
    return row


def _attach_quality_metrics(run_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["problem"]), int(row.get("problem_n_obj") or 0))].append(row)

    igd_indicator_cache: dict[str, Any] = {}
    for (problem_key, n_obj), rows in grouped.items():
        reference_front = _load_reference_front(problem_key, int(n_obj))
        fronts = [np.asarray(row["_front"], dtype=float) for row in rows if np.asarray(row["_front"]).size > 0]
        if reference_front is not None:
            fronts.append(reference_front)
        ref_point = compute_hv_reference(fronts)
        if reference_front is not None and has_moocore():
            igd_indicator_cache[problem_key] = get_indicator("igd_plus", reference_front=reference_front)
        for row in rows:
            front = np.asarray(row["_front"], dtype=float)
            row["hv"] = hypervolume(front, ref_point) if front.size > 0 else 0.0
            row["hv_ref_point"] = ref_point.tolist()
            if problem_key in igd_indicator_cache and front.size > 0:
                row["igd_plus"] = float(igd_indicator_cache[problem_key].compute(front).value)
            else:
                row["igd_plus"] = None


def _build_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row.get("suite")) if row.get("suite") is not None else None, str(row["host"]), str(row["problem"]), str(row["variant"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    baseline_lookup: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    for (suite, host, problem, variant), rows in grouped.items():
        summary = {
            "suite": suite,
            "host": host,
            "problem": problem,
            "variant": variant,
            "variant_group": rows[0]["variant_group"],
            "n_runs": len(rows),
            "mean_hv": _mean([float(row["hv"]) for row in rows]),
            "mean_igd_plus": _mean_or_none([float(row["igd_plus"]) for row in rows if row["igd_plus"] is not None]),
            "mean_time_ms": _mean([float(row["time_ms"]) for row in rows]),
            "mean_average_reward": _mean([float(row["average_bounded_reward"]) for row in rows if row["average_bounded_reward"] is not None]),
            "mean_average_overhead_ms": _mean([float(row["average_overhead_ms"]) for row in rows if row["average_overhead_ms"] is not None]),
            "mean_family_concentration": _mean([float(row["family_concentration"]) for row in rows if row["family_concentration"] is not None]),
            "mean_regime_concentration": _mean([float(row["regime_concentration"]) for row in rows if row["regime_concentration"] is not None]),
            "mean_intent_concentration": _mean([float(row["intent_concentration"]) for row in rows if row["intent_concentration"] is not None]),
            "mean_family_switches": _mean([float(row["family_switches"]) for row in rows]),
            "mean_intent_switches": _mean([float(row["intent_switches"]) for row in rows]),
            "mean_regime_switches": _mean([float(row["regime_switches"]) for row in rows]),
        }
        for profile_key in (
            "profile_controller_time_ms",
            "profile_start_step_time_ms",
            "profile_router_time_ms",
            "profile_policy_select_time_ms",
            "profile_policy_update_time_ms",
            "profile_trace_time_ms",
            "profile_decode_time_ms",
            "profile_variation_time_ms",
            "profile_evaluation_time_ms",
            "profile_survival_time_ms",
            "profile_total_runtime_ms",
        ):
            values = [float(row[profile_key]) for row in rows if row.get(profile_key) is not None]
            summary[f"mean_{profile_key}"] = _mean(values)
        for prefix, labels in (
            ("family_share", FAMILY_LABELS),
            ("regime_share", REGIME_LABELS),
            ("intent_share", INTENT_LABELS),
        ):
            for label in labels:
                key = f"{prefix}_{label}"
                summary[f"mean_{key}"] = _mean([float(row[key]) for row in rows])
        summary_rows.append(summary)
        if variant == BASELINE_BY_HOST.get(host):
            baseline_lookup[(suite, host, problem)] = summary

    hv_tolerance = 1e-6
    for row in summary_rows:
        suite = str(row.get("suite")) if row.get("suite") is not None else None
        baseline = baseline_lookup.get((suite, str(row["host"]), str(row["problem"])))
        if baseline is None:
            row["comparison_to_baseline"] = "no_baseline"
            row["hv_delta_vs_baseline"] = None
            row["runtime_ratio_vs_baseline"] = None
            continue
        if row["variant"] == baseline["variant"]:
            row["comparison_to_baseline"] = "baseline"
            row["hv_delta_vs_baseline"] = 0.0
            row["runtime_ratio_vs_baseline"] = 1.0
            continue
        hv_delta = float(row["mean_hv"]) - float(baseline["mean_hv"])
        if hv_delta > hv_tolerance:
            comparison = "win"
        elif hv_delta < -hv_tolerance:
            comparison = "loss"
        else:
            comparison = "tie"
        row["comparison_to_baseline"] = comparison
        row["hv_delta_vs_baseline"] = hv_delta
        baseline_time = max(1e-9, float(baseline["mean_time_ms"]))
        row["runtime_ratio_vs_baseline"] = float(row["mean_time_ms"]) / baseline_time
    summary_rows.sort(key=lambda item: (str(item.get("suite") or ""), item["host"], item["problem"], item["variant"]))
    return summary_rows


def _build_go_no_go(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive_rows = [row for row in summary_rows if str(row.get("variant_group")) == "adaptive"]
    wins = sum(1 for row in adaptive_rows if row.get("comparison_to_baseline") == "win")
    losses = sum(1 for row in adaptive_rows if row.get("comparison_to_baseline") == "loss")
    ties = sum(1 for row in adaptive_rows if row.get("comparison_to_baseline") == "tie")
    efficient_wins = [
        row
        for row in adaptive_rows
        if row.get("comparison_to_baseline") == "win" and float(row.get("runtime_ratio_vs_baseline") or 0.0) <= 1.25
    ]
    decision = "go" if efficient_wins and wins >= losses else "no_go"

    best_by_host: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["host"])].append(row)
    for host, rows in grouped.items():
        best = max(rows, key=lambda item: float(item["mean_hv"]))
        best_by_host[host] = {
            "variant": best["variant"],
            "problem": best["problem"],
            "mean_hv": best["mean_hv"],
            "runtime_ratio_vs_baseline": best.get("runtime_ratio_vs_baseline"),
        }

    return {
        "decision": decision,
        "baseline_by_host": BASELINE_BY_HOST,
        "adaptive_wins": wins,
        "adaptive_losses": losses,
        "adaptive_ties": ties,
        "efficient_adaptive_wins": len(efficient_wins),
        "best_by_host": best_by_host,
        "notes": [
            "Decision is GO when at least one adaptive semantic variant beats the host baseline without >25% mean runtime inflation.",
            "HV is used for baseline comparison; IGD+ is attached when MooCore is available and a reference front exists.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small semantic online-control pilot study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config file for the pilot study.")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "online_control_pilot", help="Output directory for CSV/JSON artifacts.")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    n_var = int(config.get("n_var", 10))
    hosts = [str(host) for host in config.get("hosts", ["nsgaii", "moead"])]
    problem_specs = _normalize_problem_specs(list(config.get("problems", ["zdt1", "zdt2"])), n_var)
    seeds = [int(seed) for seed in config.get("seeds", [0, 1])]
    variants = [str(variant) for variant in config.get("variants", [])]
    engine = str(config.get("engine", "numpy"))
    pop_size = int(config.get("population_size", 32))
    max_evaluations = int(config.get("max_evaluations", 320))
    credit_model = str(config.get("credit_model", "simple_improvement"))

    run_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    policy_state_dir = output_dir / "policy_states"
    for host in hosts:
        for problem_spec in problem_specs:
            problem_key = str(problem_spec["key"])
            problem_n_var = int(problem_spec["n_var"])
            problem_n_obj = int(problem_spec["n_obj"]) if problem_spec.get("n_obj") is not None else None
            suite = str(problem_spec["suite"]) if problem_spec.get("suite") is not None else None
            for seed in seeds:
                for variant in variants:
                    print(f"[pilot] host={host} problem={problem_key} seed={seed} variant={variant}")
                    run = _run_variant(
                        host=host,
                        variant=variant,
                        problem_key=problem_key,
                        n_var=problem_n_var,
                        n_obj=problem_n_obj,
                        suite=suite,
                        pop_size=pop_size,
                        max_evaluations=max_evaluations,
                        seed=seed,
                        engine=engine,
                        credit_model=credit_model,
                    )
                    policy_state = run.get("_policy_state")
                    if isinstance(policy_state, dict):
                        policy_state_dir.mkdir(parents=True, exist_ok=True)
                        policy_state_path = policy_state_dir / f"{run['run_id']}.json"
                        policy_state_path.write_text(json.dumps(policy_state, indent=2, sort_keys=True), encoding="utf-8")
                        run["policy_state_file"] = str(policy_state_path.relative_to(output_dir))
                    raw_trace_rows = run.get("_trace_rows", [])
                    if isinstance(raw_trace_rows, list):
                        for step in raw_trace_rows:
                            if not isinstance(step, dict):
                                continue
                            payload = {
                                "run_id": run["run_id"],
                                "host": run["host"],
                                "problem": run["problem"],
                                "suite": run.get("suite"),
                                "variant": run["variant"],
                                "variant_group": run["variant_group"],
                                "seed": run["seed"],
                            }
                            payload.update(step)
                            trace_rows.append(payload)
                    run_rows.append(run)

    _attach_quality_metrics(run_rows)
    summary_rows = _build_summary(run_rows)
    go_no_go = _build_go_no_go(summary_rows)

    serializable_runs = []
    for row in run_rows:
        payload = {key: value for key, value in row.items() if key not in {"_front", "_trace_rows", "_policy_state"}}
        serializable_runs.append(payload)

    _write_csv(output_dir / "runs.csv", serializable_runs)
    _write_csv(output_dir / "trace_rows.csv", trace_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "go_no_go_summary.json").write_text(json.dumps(go_no_go, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[pilot] wrote {output_dir / 'runs.csv'}")
    print(f"[pilot] wrote {output_dir / 'trace_rows.csv'}")
    print(f"[pilot] wrote {output_dir / 'summary.csv'}")
    print(f"[pilot] wrote {output_dir / 'go_no_go_summary.json'}")
    print(f"[pilot] decision={go_no_go['decision']}")


if __name__ == "__main__":
    main()

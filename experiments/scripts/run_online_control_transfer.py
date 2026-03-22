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
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.moocore_indicators import get_indicator, has_moocore
from vamos.resources import weight_path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "online_control_transfer.json"


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


def _build_online_control_payload(credit_model: str, policy_state: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": True,
        "policy": "adaptive_hierarchical_joint",
        "credit_model": credit_model,
        "trace_level": "basic",
    }
    if policy_state is not None:
        payload["policy_state"] = policy_state
    return payload


def _build_nsgaii_config(*, pop_size: int, n_var: int, credit_model: str, policy_state: dict[str, Any] | None = None) -> NSGAIIConfig:
    mut_prob = 1.0 / float(n_var)
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=mut_prob, eta=20.0)
        .selection("tournament", size=2)
        .result_mode("non_dominated")
        .online_control(_build_online_control_payload(credit_model, policy_state))
        .build()
    )


def _build_moead_config(
    *,
    pop_size: int,
    n_var: int,
    n_obj: int,
    credit_model: str,
    policy_state: dict[str, Any] | None = None,
) -> MOEADConfig:
    mut_prob = 1.0 / float(n_var)
    builder = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .batch_size(1)
        .neighbor_size(min(10, pop_size))
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=mut_prob, eta=20.0)
        .aggregation("pbi", theta=5.0)
        .result_mode("non_dominated")
        .online_control(_build_online_control_payload(credit_model, policy_state))
    )
    if n_obj > 2:
        builder.weight_vectors(path=str(weight_path("W3D_91.dat").parent))
    return builder.build()


def _build_algorithm_config(
    host: str,
    *,
    pop_size: int,
    n_var: int,
    n_obj: int,
    credit_model: str,
    policy_state: dict[str, Any] | None = None,
) -> Any:
    if host == "nsgaii":
        return _build_nsgaii_config(pop_size=pop_size, n_var=n_var, credit_model=credit_model, policy_state=policy_state)
    if host == "moead":
        return _build_moead_config(
            pop_size=pop_size,
            n_var=n_var,
            n_obj=n_obj,
            credit_model=credit_model,
            policy_state=policy_state,
        )
    raise ValueError(f"Unsupported host '{host}'.")


def _load_reference_front(problem_key: str, n_obj: int) -> np.ndarray | None:
    front_path = resolve_reference_front_path(problem_key, None, n_obj=n_obj)
    if front_path is None:
        return None
    arr = np.loadtxt(front_path, delimiter=",")
    return np.atleast_2d(np.asarray(arr, dtype=float))


def _run(
    *,
    host: str,
    problem_key: str,
    n_var: int,
    pop_size: int,
    max_evaluations: int,
    seed: int,
    engine: str,
    credit_model: str,
    policy_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection = make_problem_selection(problem_key, n_var=n_var)
    problem = selection.instantiate()
    cfg = _build_algorithm_config(
        host,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=selection.n_obj,
        credit_model=credit_model,
        policy_state=policy_state,
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
    online = result.data.get("online_control", {})
    run_summary = online.get("run_summary", {}) if isinstance(online, dict) else {}
    return {
        "host": host,
        "problem": problem_key,
        "seed": seed,
        "time_ms": elapsed_ms,
        "average_bounded_reward": run_summary.get("average_bounded_reward"),
        "policy_state": online.get("policy_state") if isinstance(online, dict) else None,
        "_front": np.asarray(result.F, dtype=float) if result.F is not None else np.empty((0, selection.n_obj), dtype=float),
    }


def _attach_quality(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem"])].append(row)
    for problem_key, bucket in grouped.items():
        n_obj = bucket[0]["_front"].shape[1] if bucket else 2
        fronts = [np.asarray(row["_front"], dtype=float) for row in bucket if np.asarray(row["_front"]).size > 0]
        reference_front = _load_reference_front(problem_key, int(n_obj))
        if reference_front is not None:
            fronts.append(reference_front)
        ref_point = compute_hv_reference(fronts)
        igd_indicator = get_indicator("igd_plus", reference_front=reference_front) if reference_front is not None and has_moocore() else None
        for row in bucket:
            front = np.asarray(row["_front"], dtype=float)
            row["hv"] = hypervolume(front, ref_point) if front.size > 0 else 0.0
            row["igd_plus"] = float(igd_indicator.compute(front).value) if igd_indicator is not None and front.size > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny cross-host online-control transfer study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config file for transfer experiments.")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "online_control_transfer", help="Output directory.")
    args = parser.parse_args()

    config = _load_config(args.config)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_state_dir = output_dir / "policy_states"

    directions = [(str(item[0]), str(item[1])) for item in config.get("directions", [["nsgaii", "moead"], ["moead", "nsgaii"]])]
    problems = [str(problem) for problem in config.get("problems", ["zdt1"])]
    seeds = [int(seed) for seed in config.get("seeds", [0, 1])]
    n_var = int(config.get("n_var", 10))
    pop_size = int(config.get("population_size", 24))
    source_budget = int(config.get("source_max_evaluations", 160))
    target_budget = int(config.get("target_max_evaluations", 160))
    engine = str(config.get("engine", "numpy"))
    credit_model = str(config.get("credit_model", "simple_improvement"))

    transfer_runs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for source_host, target_host in directions:
        for problem_key in problems:
            for seed in seeds:
                print(f"[transfer] source={source_host} target={target_host} problem={problem_key} seed={seed}")
                source = _run(
                    host=source_host,
                    problem_key=problem_key,
                    n_var=n_var,
                    pop_size=pop_size,
                    max_evaluations=source_budget,
                    seed=seed,
                    engine=engine,
                    credit_model=credit_model,
                )
                policy_state = source["policy_state"] if isinstance(source.get("policy_state"), dict) else None
                if policy_state is not None:
                    policy_state_dir.mkdir(parents=True, exist_ok=True)
                    state_path = policy_state_dir / f"{source_host}_to_{target_host}__{problem_key}__seed{seed}.json"
                    state_path.write_text(json.dumps(policy_state, indent=2, sort_keys=True), encoding="utf-8")
                cold = _run(
                    host=target_host,
                    problem_key=problem_key,
                    n_var=n_var,
                    pop_size=pop_size,
                    max_evaluations=target_budget,
                    seed=seed,
                    engine=engine,
                    credit_model=credit_model,
                )
                warm = _run(
                    host=target_host,
                    problem_key=problem_key,
                    n_var=n_var,
                    pop_size=pop_size,
                    max_evaluations=target_budget,
                    seed=seed,
                    engine=engine,
                    credit_model=credit_model,
                    policy_state=policy_state,
                )
                rows = [
                    {"direction": f"{source_host}_to_{target_host}", "role": "source", "mode": "source", "source_host": source_host, "target_host": target_host, "problem": problem_key, "seed": seed, **source},
                    {"direction": f"{source_host}_to_{target_host}", "role": "target", "mode": "cold", "source_host": source_host, "target_host": target_host, "problem": problem_key, "seed": seed, **cold},
                    {"direction": f"{source_host}_to_{target_host}", "role": "target", "mode": "warm", "source_host": source_host, "target_host": target_host, "problem": problem_key, "seed": seed, **warm},
                ]
                _attach_quality(rows)
                for row in rows:
                    transfer_runs.append({key: value for key, value in row.items() if key not in {"policy_state", "_front"}})
                hv_delta = float(rows[2]["hv"]) - float(rows[1]["hv"])
                igd_delta = (
                    float(rows[2]["igd_plus"]) - float(rows[1]["igd_plus"])
                    if rows[2].get("igd_plus") is not None and rows[1].get("igd_plus") is not None
                    else None
                )
                summary_rows.append(
                    {
                        "direction": f"{source_host}_to_{target_host}",
                        "source_host": source_host,
                        "target_host": target_host,
                        "problem": problem_key,
                        "seed": seed,
                        "hv_delta_warm_vs_cold": hv_delta,
                        "igd_plus_delta_warm_vs_cold": igd_delta,
                        "runtime_ratio_warm_vs_cold": float(rows[2]["time_ms"]) / max(1e-9, float(rows[1]["time_ms"])),
                        "reward_delta_warm_vs_cold": (
                            float(rows[2]["average_bounded_reward"]) - float(rows[1]["average_bounded_reward"])
                            if rows[2].get("average_bounded_reward") is not None and rows[1].get("average_bounded_reward") is not None
                            else None
                        ),
                        "outcome": "win" if hv_delta > 1e-6 else "loss" if hv_delta < -1e-6 else "tie",
                    }
                )

    _write_csv(output_dir / "transfer_runs.csv", transfer_runs)
    _write_csv(output_dir / "transfer_summary.csv", summary_rows)
    print(f"[transfer] wrote {output_dir / 'transfer_runs.csv'}")
    print(f"[transfer] wrote {output_dir / 'transfer_summary.csv'}")


if __name__ == "__main__":
    main()

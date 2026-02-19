#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

try:  # Reduce noise when pymoo is installed without compiled extensions.
    from pymoo.config import Config as _PymooConfig

    _PymooConfig.warnings["not_compiled"] = False
except Exception:
    pass


SUPPORTED_ALGORITHMS = ("nsgaii", "moead", "smsemoa")
SUPPORTED_FRAMEWORKS = ("vamos", "jmetalpy", "pymoo")
METRIC_DIRECTIONS = {
    "igd_plus": "min",
    "epsilon_additive": "min",
    "hv_normalized": "max",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    complicated_pareto_set: bool
    level: int
    bias: bool
    imbalance: bool


@dataclass(frozen=True)
class Task:
    framework: str
    algorithm: str
    problem_id: int
    n_obj: int
    n_var: int
    scenario: Scenario
    seed: int
    population_size: int
    max_evaluations: int


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _print(msg: str) -> None:
    print(f"[{_now()}] {msg}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required. Install with `pip install pyyaml`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_seed_spec(seed_spec: Any) -> list[int]:
    if isinstance(seed_spec, list):
        values = [int(v) for v in seed_spec]
        if not values:
            raise ValueError("seeds list cannot be empty.")
        return values
    if isinstance(seed_spec, dict):
        start = int(seed_spec.get("start", 1))
        count = int(seed_spec.get("count", 1))
        step = int(seed_spec.get("step", 1))
        if count <= 0:
            raise ValueError("seeds.count must be > 0.")
        return [start + i * step for i in range(count)]
    raise TypeError("experiment.seeds must be a list[int] or mapping {start,count,step}.")


def _parse_cli_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        return None
    return [int(v) for v in values]


def _parse_cli_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        return None
    return values


def _parse_datetime_local(raw: str) -> datetime:
    text = str(raw).strip()
    if not text:
        raise ValueError("Deadline timestamp cannot be empty.")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        # Convert aware datetime to local time, then compare as naive local datetime.
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _get_run_deadline(manifest: dict[str, Any]) -> datetime | None:
    deadline_cfg = manifest.get("deadline", {})
    if not isinstance(deadline_cfg, dict):
        raise ValueError("Manifest 'deadline' must be a mapping when provided.")
    raw = deadline_cfg.get("stop_new_runs_at")
    if raw in (None, ""):
        return None
    return _parse_datetime_local(str(raw))


def _resolve_workers(manifest: dict[str, Any], cli_workers: int | None) -> int:
    if cli_workers is not None:
        raw = int(cli_workers)
    else:
        run_cfg = manifest.get("run", {})
        if isinstance(run_cfg, dict) and "workers" in run_cfg:
            raw = int(run_cfg["workers"])
        else:
            raw = 1

    if raw == -1:
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    if raw < 1:
        raise ValueError("workers must be >= 1 or -1 (for cpu_count()-1).")
    return raw


def _add_repo_paths(paths_cfg: dict[str, Any]) -> None:
    vamos_root = Path(paths_cfg["vamos_root"])
    jmetalpy_root = Path(paths_cfg["jmetalpy_root"])
    pymoo_root = Path(paths_cfg["pymoo_root"])

    candidate_paths = [
        vamos_root / "src",
        jmetalpy_root / "src",
        pymoo_root,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            sys.path.insert(0, str(candidate))


def _manifest_and_paths(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = _load_yaml(manifest_path)
    base = manifest_path.parent
    paths = manifest.get("paths", {})
    required = (
        "vamos_root",
        "jmetalpy_root",
        "pymoo_root",
        "reference_front_dir",
        "jmetalpy_reference_front_dir",
        "output_dir",
        "moead_weight_dir",
    )
    missing = [key for key in required if key not in paths]
    if missing:
        raise ValueError(f"Manifest paths missing required keys: {missing}")

    resolved = {key: _resolve_path(base, str(paths[key])) for key in required}
    _ensure_dir(resolved["output_dir"])
    _ensure_dir(resolved["moead_weight_dir"])
    _add_repo_paths(resolved)
    return manifest, resolved


def _build_scenarios(manifest: dict[str, Any]) -> list[Scenario]:
    scenario_rows = manifest["experiment"]["scenarios"]
    scenarios: list[Scenario] = []
    for row in scenario_rows:
        scenarios.append(
            Scenario(
                scenario_id=str(row["id"]),
                complicated_pareto_set=bool(row.get("complicated_pareto_set", False)),
                level=int(row.get("level", 1)),
                bias=bool(row.get("bias", False)),
                imbalance=bool(row.get("imbalance", False)),
            )
        )
    if not scenarios:
        raise ValueError("experiment.scenarios cannot be empty.")
    return scenarios


def _build_tasks(
    manifest: dict[str, Any],
    *,
    frameworks: list[str] | None = None,
    algorithms: list[str] | None = None,
    problem_ids: list[int] | None = None,
    objective_counts: list[int] | None = None,
    scenario_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[Task]:
    exp = manifest["experiment"]
    fw_list = [str(v).lower() for v in (frameworks or exp["frameworks"])]
    algo_list = [str(v).lower() for v in (algorithms or exp["algorithms"])]
    pid_list = [int(v) for v in (problem_ids or exp["problem_ids"])]
    n_obj_list = [int(v) for v in (objective_counts or exp["objective_counts"])]
    all_scenarios = _build_scenarios(manifest)
    if scenario_ids:
        keep = {v for v in scenario_ids}
        scenarios = [s for s in all_scenarios if s.scenario_id in keep]
    else:
        scenarios = all_scenarios
    if not scenarios:
        raise ValueError("No scenarios selected.")
    seed_list = seeds or _parse_seed_spec(exp["seeds"])

    for fw in fw_list:
        if fw not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework '{fw}'. Supported: {SUPPORTED_FRAMEWORKS}")
    for algo in algo_list:
        if algo not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algo}'. Supported: {SUPPORTED_ALGORITHMS}")

    n_var = int(exp["n_var"])
    pop_size = int(exp["population_size"])
    eval_factor = int(exp["max_evaluations_per_objective"])

    tasks: list[Task] = []
    for scenario in scenarios:
        for n_obj in n_obj_list:
            max_eval = eval_factor * n_obj
            for pid in pid_list:
                for algo in algo_list:
                    for fw in fw_list:
                        for seed in seed_list:
                            tasks.append(
                                Task(
                                    framework=fw,
                                    algorithm=algo,
                                    problem_id=pid,
                                    n_obj=n_obj,
                                    n_var=n_var,
                                    scenario=scenario,
                                    seed=int(seed),
                                    population_size=pop_size,
                                    max_evaluations=max_eval,
                                )
                            )
    return tasks


def _bounds(n_var: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(1.0, n_var + 1.0, dtype=float)
    return -0.5 * idx, 0.5 * idx


def _nondominated(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return F.reshape(0, 0)
    F = np.asarray(F, dtype=float)
    keep = np.ones(F.shape[0], dtype=bool)
    for i in range(F.shape[0]):
        if not keep[i]:
            continue
        dominates = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        keep[dominates] = False
    out = F[keep]
    if out.size == 0:
        return out.reshape(0, F.shape[1])
    out = np.unique(out, axis=0)
    return out


def _read_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean:
            continue
        tokens = re.split(r"[,\s]+", clean)
        rows.append([float(tok) for tok in tokens if tok != ""])
    if not rows:
        raise ValueError(f"Empty matrix file: {path}")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent row width in {path}")
    return np.asarray(rows, dtype=float)


def _ref_front_path(reference_front_dir: Path, problem_id: int, n_obj: int) -> Path:
    if n_obj == 2:
        preferred = reference_front_dir / f"zcat{problem_id}.2d.csv"
        if preferred.exists():
            return preferred
        fallback = reference_front_dir / f"zcat{problem_id}.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Reference front not found for ZCAT{problem_id} 2D.")
    path = reference_front_dir / f"zcat{problem_id}.{n_obj}d.csv"
    if not path.exists():
        raise FileNotFoundError(f"Reference front not found: {path}")
    return path


def _jmetalpy_ref_front_path(reference_front_dir: Path, problem_id: int, n_obj: int) -> Path:
    if n_obj == 2:
        name = f"ZCAT{problem_id}.2D.pf"
        alt = f"ZCAT{problem_id}.pf"
        if (reference_front_dir / name).exists():
            return reference_front_dir / name
        path_alt = reference_front_dir / alt
        if path_alt.exists():
            return path_alt
        raise FileNotFoundError(f"jMetalPy reference front not found for ZCAT{problem_id} 2D.")
    path = reference_front_dir / f"ZCAT{problem_id}.{n_obj}D.pf"
    if not path.exists():
        raise FileNotFoundError(f"jMetalPy reference front not found: {path}")
    return path


def _instantiate_problem(framework: str, task: Task) -> Any:
    pid = task.problem_id
    s = task.scenario
    if framework == "vamos":
        from vamos.foundation.problem import zcat as vamos_zcat

        cls = getattr(vamos_zcat, f"ZCAT{pid}Problem")
        return cls(
            n_var=task.n_var,
            n_obj=task.n_obj,
            complicated_pareto_set=s.complicated_pareto_set,
            level=s.level,
            bias=s.bias,
            imbalance=s.imbalance,
        )
    if framework == "jmetalpy":
        import jmetal.problem as jmetal_problem

        cls = getattr(jmetal_problem, f"ZCAT{pid}")
        return cls(
            number_of_variables=task.n_var,
            number_of_objectives=task.n_obj,
            complicated_pareto_set=s.complicated_pareto_set,
            level=s.level,
            bias=s.bias,
            imbalance=s.imbalance,
        )
    if framework == "pymoo":
        from pymoo.problems.many import zcat as pymoo_zcat

        cls = getattr(pymoo_zcat, f"ZCAT{pid}")
        return cls(
            n_var=task.n_var,
            n_obj=task.n_obj,
            complicated_pareto_set=s.complicated_pareto_set,
            level=s.level,
            bias=s.bias,
            imbalance=s.imbalance,
        )
    raise ValueError(f"Unknown framework: {framework}")


def _evaluate_problem(framework: str, problem: Any, X: np.ndarray, n_obj: int) -> np.ndarray:
    if framework == "vamos":
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)
        return np.asarray(out["F"], dtype=float)
    if framework == "jmetalpy":
        F = np.empty((X.shape[0], n_obj), dtype=float)
        for i, row in enumerate(X):
            sol = problem.create_solution()
            sol.variables = [float(v) for v in row]
            problem.evaluate(sol)
            F[i, :] = np.asarray(sol.objectives, dtype=float)
        return F
    if framework == "pymoo":
        return np.asarray(problem.evaluate(X, return_values_of=["F"]), dtype=float)
    raise ValueError(f"Unknown framework: {framework}")


def _count_lattice_points(n_obj: int, divisions: int) -> int:
    return math.comb(divisions + n_obj - 1, n_obj - 1)


def _choose_divisions(pop_size: int, n_obj: int) -> int:
    divisions = 1
    while _count_lattice_points(n_obj, divisions) < pop_size:
        divisions += 1
    return divisions


def _simplex_lattice(n_obj: int, divisions: int, limit: int) -> np.ndarray:
    coords: list[tuple[int, ...]] = []

    def rec(remaining: int, depth: int, current: list[int]) -> None:
        if len(coords) >= limit:
            return
        if depth == n_obj - 1:
            current.append(remaining)
            coords.append(tuple(current))
            current.pop()
            return
        for value in range(remaining + 1):
            current.append(value)
            rec(remaining - value, depth + 1, current)
            current.pop()
            if len(coords) >= limit:
                return

    rec(divisions, 0, [])
    weights = np.asarray(coords, dtype=float)
    weights /= float(divisions)
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _moead_weights(pop_size: int, n_obj: int) -> np.ndarray:
    if n_obj == 2:
        values = np.linspace(0.0, 1.0, pop_size, dtype=float)
        return np.column_stack([values, 1.0 - values])
    divisions = _choose_divisions(pop_size, n_obj)
    return _simplex_lattice(n_obj, divisions, pop_size)


def _ensure_weight_files(weight_dir: Path, pop_size: int, n_obj: int) -> tuple[np.ndarray, Path]:
    _ensure_dir(weight_dir)
    stem = f"W{n_obj}D_{pop_size}"
    dat_path = weight_dir / f"{stem}.dat"
    csv_path = weight_dir / f"{stem}.csv"
    if dat_path.exists() and csv_path.exists():
        return _read_matrix(dat_path), dat_path
    weights = _moead_weights(pop_size, n_obj)
    np.savetxt(dat_path, weights, fmt="%.18e")
    np.savetxt(csv_path, weights, delimiter=",", fmt="%.18e")
    return weights, dat_path


def _run_vamos(task: Task, engine: str, weight_dir: Path) -> np.ndarray:
    from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig, SMSEMOAConfig
    from vamos.experiment.unified import optimize

    problem = _instantiate_problem("vamos", task)

    if task.algorithm == "nsgaii":
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(task.population_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=1.0 / task.n_var, eta=20.0)
            .selection("tournament")
            .build()
        )
    elif task.algorithm == "smsemoa":
        cfg = (
            SMSEMOAConfig.builder()
            .pop_size(task.population_size)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob=1.0 / task.n_var, eta=20.0)
            .selection("random")
            .reference_point(adaptive=True, offset=1.0)
            .build()
        )
    elif task.algorithm == "moead":
        _, weight_path = _ensure_weight_files(weight_dir, task.population_size, task.n_obj)
        cfg = (
            MOEADConfig.builder()
            .pop_size(task.population_size)
            .batch_size(1)
            .neighbor_size(20)
            .delta(0.9)
            .replace_limit(2)
            .crossover("de", cr=1.0, f=0.5)
            .mutation("pm", prob=1.0 / task.n_var, eta=20.0)
            .aggregation("tchebycheff")
            .weight_vectors(path=str(weight_path.parent))
            .build()
        )
    else:
        raise ValueError(f"Unsupported algorithm for VAMOS: {task.algorithm}")

    result = optimize(
        problem,
        algorithm=task.algorithm,
        algorithm_config=cfg,
        max_evaluations=task.max_evaluations,
        seed=task.seed,
        engine=engine,
        verbose=False,
    )
    return np.asarray(result.F, dtype=float)


def _run_jmetalpy(task: Task, weight_dir: Path) -> np.ndarray:
    from jmetal.algorithm.multiobjective.moead import MOEAD
    from jmetal.algorithm.multiobjective.nsgaii import NSGAII
    from jmetal.algorithm.multiobjective.smsemoa import SMSEMOA
    from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
    from jmetal.operator.mutation import PolynomialMutation
    from jmetal.util.aggregation_function import Tschebycheff
    from jmetal.util.termination_criterion import StoppingByEvaluations

    import random

    random.seed(task.seed)
    np.random.seed(task.seed)
    logging.getLogger("jmetal").setLevel(logging.WARNING)
    logging.getLogger("jmetal.core.algorithm").setLevel(logging.WARNING)

    problem = _instantiate_problem("jmetalpy", task)
    mutation = PolynomialMutation(probability=1.0 / task.n_var, distribution_index=20.0)
    termination = StoppingByEvaluations(max_evaluations=task.max_evaluations)

    if task.algorithm == "nsgaii":
        algorithm = NSGAII(
            problem=problem,
            population_size=task.population_size,
            offspring_population_size=task.population_size,
            mutation=mutation,
            crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
            termination_criterion=termination,
        )
    elif task.algorithm == "smsemoa":
        algorithm = SMSEMOA(
            problem=problem,
            population_size=task.population_size,
            mutation=mutation,
            crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
            termination_criterion=termination,
        )
    elif task.algorithm == "moead":
        _ensure_weight_files(weight_dir, task.population_size, task.n_obj)
        algorithm = MOEAD(
            problem=problem,
            population_size=task.population_size,
            mutation=mutation,
            crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
            aggregation_function=Tschebycheff(task.n_obj),
            neighbourhood_selection_probability=0.9,
            max_number_of_replaced_solutions=2,
            neighbor_size=20,
            weight_files_path=str(weight_dir),
            termination_criterion=termination,
        )
    else:
        raise ValueError(f"Unsupported algorithm for jMetalPy: {task.algorithm}")

    algorithm.run()
    result_fn = getattr(algorithm, "result", None)
    solutions = result_fn() if callable(result_fn) else []
    if not solutions:
        return np.empty((0, task.n_obj), dtype=float)
    return np.asarray([sol.objectives for sol in solutions], dtype=float)


def _run_pymoo(task: Task, weights: np.ndarray | None) -> np.ndarray:
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.operators.crossover.dex import DEX
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize

    problem = _instantiate_problem("pymoo", task)

    if task.algorithm == "nsgaii":
        algorithm = NSGA2(
            pop_size=task.population_size,
            crossover=SBX(prob=0.9, eta=20.0),
            mutation=PM(prob=1.0, prob_var=1.0 / task.n_var, eta=20.0),
        )
    elif task.algorithm == "smsemoa":
        algorithm = SMSEMOA(
            pop_size=task.population_size,
            crossover=SBX(prob=0.9, eta=20.0),
            mutation=PM(prob=1.0, prob_var=1.0 / task.n_var, eta=20.0),
            normalize=False,
            eliminate_duplicates=False,
        )
    elif task.algorithm == "moead":
        if weights is None:
            raise ValueError("MOEA/D in pymoo requires explicit reference directions.")
        algorithm = MOEAD(
            ref_dirs=weights,
            n_neighbors=20,
            decomposition=Tchebicheff(),
            prob_neighbor_mating=0.9,
            crossover=DEX(F=0.5, CR=1.0, variant="bin", jitter=False),
            mutation=PM(prob=1.0, prob_var=1.0 / task.n_var, eta=20.0),
        )
    else:
        raise ValueError(f"Unsupported algorithm for pymoo: {task.algorithm}")

    result = minimize(
        problem,
        algorithm,
        ("n_eval", task.max_evaluations),
        seed=task.seed,
        verbose=False,
    )
    if result.F is None:
        return np.empty((0, task.n_obj), dtype=float)
    return np.asarray(result.F, dtype=float)


def _run_task(task: Task, framework_options: dict[str, Any], weight_dir: Path) -> np.ndarray:
    if task.framework == "vamos":
        engine = str(framework_options.get("vamos", {}).get("engine", "numpy"))
        return _run_vamos(task, engine=engine, weight_dir=weight_dir)
    if task.framework == "jmetalpy":
        return _run_jmetalpy(task, weight_dir=weight_dir)
    if task.framework == "pymoo":
        weights = None
        if task.algorithm == "moead":
            weights, _ = _ensure_weight_files(weight_dir, task.population_size, task.n_obj)
        return _run_pymoo(task, weights=weights)
    raise ValueError(f"Unsupported framework: {task.framework}")


def _run_one_task_worker(
    task: Task,
    framework_options: dict[str, Any],
    weight_dir_raw: str,
    output_dir_raw: str,
    repo_paths: dict[str, str],
) -> dict[str, Any]:
    # Worker processes started with spawn (Windows) need explicit path wiring.
    _add_repo_paths(repo_paths)

    weight_dir = Path(weight_dir_raw)
    output_dir = Path(output_dir_raw)

    start = time.perf_counter()
    status = "ok"
    error: str | None = None
    F = np.empty((0, task.n_obj), dtype=float)
    try:
        F = _run_task(task, framework_options=framework_options, weight_dir=weight_dir)
    except Exception as exc:  # pragma: no cover - runtime guard
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start
    return _save_run(task, output_dir=output_dir, F_raw=F, runtime_seconds=runtime, status=status, error=error)


def _task_run_dir(output_dir: Path, task: Task) -> Path:
    return (
        output_dir
        / "raw"
        / task.scenario.scenario_id
        / f"m{task.n_obj}"
        / f"zcat{task.problem_id}"
        / task.algorithm
        / task.framework
        / f"seed_{task.seed}"
    )


def _save_run(task: Task, output_dir: Path, F_raw: np.ndarray, runtime_seconds: float, status: str, error: str | None) -> dict[str, Any]:
    run_dir = _task_run_dir(output_dir, task)
    _ensure_dir(run_dir)
    front_raw_path = run_dir / "front_raw.csv"
    front_nd_path = run_dir / "front.csv"
    meta_path = run_dir / "meta.json"

    if status == "ok":
        F_raw = np.asarray(F_raw, dtype=float)
        F_nd = _nondominated(F_raw)
        np.savetxt(front_raw_path, F_raw, delimiter=",", fmt="%.18e")
        np.savetxt(front_nd_path, F_nd, delimiter=",", fmt="%.18e")
        front_size = int(F_nd.shape[0])
        front_raw_size = int(F_raw.shape[0])
    else:
        front_size = 0
        front_raw_size = 0

    row = {
        "framework": task.framework,
        "algorithm": task.algorithm,
        "scenario_id": task.scenario.scenario_id,
        "problem_id": task.problem_id,
        "n_obj": task.n_obj,
        "n_var": task.n_var,
        "seed": task.seed,
        "population_size": task.population_size,
        "max_evaluations": task.max_evaluations,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "error": error or "",
        "front_size": front_size,
        "front_raw_size": front_raw_size,
        "front_path": str(front_nd_path),
        "front_raw_path": str(front_raw_path),
        "run_dir": str(run_dir),
    }

    meta = {
        "task": row,
        "scenario": {
            "id": task.scenario.scenario_id,
            "complicated_pareto_set": task.scenario.complicated_pareto_set,
            "level": task.scenario.level,
            "bias": task.scenario.bias,
            "imbalance": task.scenario.imbalance,
        },
        "timestamp": _now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return row


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _igd_plus(front: np.ndarray, ref: np.ndarray) -> float:
    if front.size == 0 or ref.size == 0:
        return float("nan")
    diff = front[None, :, :] - ref[:, None, :]
    diff = np.maximum(diff, 0.0)
    distances = np.linalg.norm(diff, axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def _epsilon_additive(front: np.ndarray, ref: np.ndarray) -> float:
    if front.size == 0 or ref.size == 0:
        return float("nan")
    diff = front[None, :, :] - ref[:, None, :]
    max_obj = np.max(diff, axis=2)
    return float(np.max(np.min(max_obj, axis=1)))


def _hv_normalized(front: np.ndarray, ref: np.ndarray, ref_point: np.ndarray) -> float:
    if front.size == 0 or ref.size == 0:
        return float("nan")
    try:
        from pymoo.indicators.hv import HV
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymoo is required for HV computation in analyze phase.") from exc
    hv = HV(ref_point=ref_point)
    hv_ref = float(hv(ref))
    if hv_ref <= 0.0:
        return float("nan")
    hv_front = float(hv(front))
    return hv_front / hv_ref


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = 0
    less = 0
    for xi in x:
        greater += int(np.sum(xi > y))
        less += int(np.sum(xi < y))
    denom = x.size * y.size
    if denom == 0:
        return float("nan")
    return (greater - less) / denom


def _holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        value = (m - rank) * p
        if value < running:
            value = running
        running = value
        adjusted[idx] = min(1.0, value)
    return adjusted


def _read_reference_front(reference_front_dir: Path, problem_id: int, n_obj: int) -> np.ndarray:
    return _read_matrix(_ref_front_path(reference_front_dir, problem_id, n_obj))


def command_validate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest, paths = _manifest_and_paths(manifest_path)

    validation_cfg = manifest.get("validation", {})
    problem_ids = _parse_cli_int_list(args.problem_ids) or [int(v) for v in validation_cfg.get("problem_ids", manifest["experiment"]["problem_ids"])]
    objective_counts = _parse_cli_int_list(args.objective_counts) or [
        int(v) for v in validation_cfg.get("objective_counts", manifest["experiment"]["objective_counts"])
    ]
    n_points = int(args.n_points or validation_cfg.get("n_random_points", 128))
    tolerance = float(args.tolerance or validation_cfg.get("tolerance", 1.0e-12))

    scenarios = _build_scenarios(manifest)
    validation_dir = paths["output_dir"] / "validation"
    _ensure_dir(validation_dir)

    _print(
        f"validate: problem_ids={len(problem_ids)}, objective_counts={objective_counts}, scenarios={len(scenarios)}, "
        f"n_points={n_points}, tolerance={tolerance}"
    )

    lower, upper = _bounds(int(manifest["experiment"]["n_var"]))
    rng = np.random.default_rng(int(args.seed))

    eq_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for n_obj in objective_counts:
            for problem_id in problem_ids:
                task = Task(
                    framework="vamos",
                    algorithm="nsgaii",
                    problem_id=problem_id,
                    n_obj=n_obj,
                    n_var=int(manifest["experiment"]["n_var"]),
                    scenario=scenario,
                    seed=0,
                    population_size=10,
                    max_evaluations=10,
                )
                X = rng.uniform(lower, upper, size=(n_points, task.n_var))
                fv = _evaluate_problem("vamos", _instantiate_problem("vamos", task), X, n_obj)
                fj = _evaluate_problem("jmetalpy", _instantiate_problem("jmetalpy", task), X, n_obj)
                fp = _evaluate_problem("pymoo", _instantiate_problem("pymoo", task), X, n_obj)

                d_vj = float(np.max(np.abs(fv - fj)))
                d_vp = float(np.max(np.abs(fv - fp)))
                d_jp = float(np.max(np.abs(fj - fp)))
                max_diff = max(d_vj, d_vp, d_jp)
                eq_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "problem_id": problem_id,
                        "n_obj": n_obj,
                        "n_points": n_points,
                        "max_abs_vamos_jmetalpy": d_vj,
                        "max_abs_vamos_pymoo": d_vp,
                        "max_abs_jmetalpy_pymoo": d_jp,
                        "max_abs_any": max_diff,
                        "pass": bool(max_diff <= tolerance),
                    }
                )

    ref_rows: list[dict[str, Any]] = []
    for n_obj in objective_counts:
        for problem_id in problem_ids:
            row: dict[str, Any] = {
                "problem_id": problem_id,
                "n_obj": n_obj,
                "pass": False,
                "shape_equal": False,
                "max_abs_diff": "",
                "error": "",
            }
            try:
                v_path = _ref_front_path(paths["reference_front_dir"], problem_id, n_obj)
                j_path = _jmetalpy_ref_front_path(paths["jmetalpy_reference_front_dir"], problem_id, n_obj)
                v_ref = _read_matrix(v_path)
                j_ref = _read_matrix(j_path)
                row["shape_equal"] = bool(v_ref.shape == j_ref.shape)
                if v_ref.shape == j_ref.shape:
                    diff = float(np.max(np.abs(v_ref - j_ref)))
                    row["max_abs_diff"] = diff
                    row["pass"] = bool(diff <= tolerance)
                else:
                    row["max_abs_diff"] = ""
                    row["pass"] = False
            except Exception as exc:  # pragma: no cover - diagnostic path
                row["error"] = str(exc)
                row["pass"] = False
            ref_rows.append(row)

    eq_path = validation_dir / "precheck_objective_equivalence.csv"
    ref_path = validation_dir / "precheck_reference_fronts.csv"
    _write_rows_csv(eq_path, eq_rows)
    _write_rows_csv(ref_path, ref_rows)

    total_eq = len(eq_rows)
    pass_eq = sum(1 for r in eq_rows if r["pass"])
    total_ref = len(ref_rows)
    pass_ref = sum(1 for r in ref_rows if r["pass"])

    summary = {
        "objective_equivalence": {"total": total_eq, "pass": pass_eq, "fail": total_eq - pass_eq},
        "reference_front_equivalence": {"total": total_ref, "pass": pass_ref, "fail": total_ref - pass_ref},
        "tolerance": tolerance,
        "n_points": n_points,
    }
    (validation_dir / "precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print(f"validate done: objective_equivalence {pass_eq}/{total_eq}, reference_fronts {pass_ref}/{total_ref}")
    _print(f"wrote: {eq_path}")
    _print(f"wrote: {ref_path}")
    return 0 if pass_eq == total_eq and pass_ref == total_ref else 1


def command_run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest, paths = _manifest_and_paths(manifest_path)
    framework_options = manifest.get("framework_options", {})
    deadline = _get_run_deadline(manifest)
    workers = _resolve_workers(manifest, args.workers)

    tasks = _build_tasks(
        manifest,
        frameworks=_parse_cli_str_list(args.frameworks),
        algorithms=_parse_cli_str_list(args.algorithms),
        problem_ids=_parse_cli_int_list(args.problem_ids),
        objective_counts=_parse_cli_int_list(args.objective_counts),
        scenario_ids=_parse_cli_str_list(args.scenarios),
        seeds=_parse_cli_int_list(args.seeds),
    )
    if args.max_runs and int(args.max_runs) > 0:
        tasks = tasks[: int(args.max_runs)]

    output_dir = paths["output_dir"]
    weight_dir = paths["moead_weight_dir"]
    _ensure_dir(output_dir)
    _ensure_dir(weight_dir)

    _print(f"run: tasks={len(tasks)}, resume={bool(args.resume)}, workers={workers}")
    if deadline is not None:
        _print(f"run deadline (local): {deadline.strftime('%Y-%m-%d %H:%M:%S')}")
    if not tasks:
        _print("No tasks selected.")
        return 0

    runs_csv = output_dir / "runs.csv"
    existing_rows: list[dict[str, Any]] = []
    if args.resume and runs_csv.exists():
        existing_rows = [dict(r) for r in _load_rows_csv(runs_csv)]

    row_pairs: list[tuple[int, dict[str, Any]]] = []
    stopped_by_deadline = False

    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            if deadline is not None and datetime.now() >= deadline:
                _print(
                    f"deadline reached; stopping before task {idx}/{len(tasks)} "
                    f"(limit={deadline.strftime('%Y-%m-%d %H:%M:%S')})"
                )
                stopped_by_deadline = True
                break

            run_dir = _task_run_dir(output_dir, task)
            meta_path = run_dir / "meta.json"
            if args.resume and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    row = dict(meta.get("task", {}))
                    row["resumed"] = True
                    row_pairs.append((idx, row))
                    _print(f"[{idx}/{len(tasks)}] resume hit: {task.framework}/{task.algorithm}/zcat{task.problem_id}/m{task.n_obj}/seed{task.seed}")
                    continue
                except Exception:
                    pass

            _print(
                f"[{idx}/{len(tasks)}] run {task.framework}/{task.algorithm} "
                f"scenario={task.scenario.scenario_id} zcat{task.problem_id} m={task.n_obj} seed={task.seed}"
            )
            start = time.perf_counter()
            status = "ok"
            error: str | None = None
            F = np.empty((0, task.n_obj), dtype=float)
            try:
                F = _run_task(task, framework_options=framework_options, weight_dir=weight_dir)
            except Exception as exc:  # pragma: no cover - runtime guard
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                _print(f"  error: {error}")
                if args.debug:
                    traceback.print_exc()
            runtime = time.perf_counter() - start
            row = _save_run(task, output_dir=output_dir, F_raw=F, runtime_seconds=runtime, status=status, error=error)
            row_pairs.append((idx, row))
    else:
        repo_paths = {
            "vamos_root": str(paths["vamos_root"]),
            "jmetalpy_root": str(paths["jmetalpy_root"]),
            "pymoo_root": str(paths["pymoo_root"]),
        }
        indexed_tasks = list(enumerate(tasks, start=1))
        next_pos = 0
        pending: dict[Future[dict[str, Any]], tuple[int, Task]] = {}

        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                while next_pos < len(indexed_tasks) and len(pending) < workers:
                    idx, task = indexed_tasks[next_pos]
                    next_pos += 1

                    if deadline is not None and datetime.now() >= deadline:
                        _print(
                            f"deadline reached; stopping before task {idx}/{len(tasks)} "
                            f"(limit={deadline.strftime('%Y-%m-%d %H:%M:%S')})"
                        )
                        stopped_by_deadline = True
                        break

                    run_dir = _task_run_dir(output_dir, task)
                    meta_path = run_dir / "meta.json"
                    if args.resume and meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            row = dict(meta.get("task", {}))
                            row["resumed"] = True
                            row_pairs.append((idx, row))
                            _print(f"[{idx}/{len(tasks)}] resume hit: {task.framework}/{task.algorithm}/zcat{task.problem_id}/m{task.n_obj}/seed{task.seed}")
                            continue
                        except Exception:
                            pass

                    _print(
                        f"[{idx}/{len(tasks)}] run {task.framework}/{task.algorithm} "
                        f"scenario={task.scenario.scenario_id} zcat{task.problem_id} m={task.n_obj} seed={task.seed}"
                    )
                    fut = pool.submit(
                        _run_one_task_worker,
                        task,
                        framework_options,
                        str(weight_dir),
                        str(output_dir),
                        repo_paths,
                    )
                    pending[fut] = (idx, task)

                if not pending:
                    if stopped_by_deadline or next_pos >= len(indexed_tasks):
                        break
                    continue

                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx, task = pending.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        error = f"{type(exc).__name__}: {exc}"
                        _print(f"  error: {error}")
                        if args.debug:
                            traceback.print_exc()
                        row = _save_run(
                            task,
                            output_dir=output_dir,
                            F_raw=np.empty((0, task.n_obj), dtype=float),
                            runtime_seconds=float("nan"),
                            status="error",
                            error=error,
                        )
                    row_pairs.append((idx, row))

                if stopped_by_deadline and not pending:
                    break

    new_rows = [row for _, row in sorted(row_pairs, key=lambda item: item[0])]

    merged = existing_rows + new_rows
    _write_rows_csv(runs_csv, merged)

    resolved_manifest_path = output_dir / "run_manifest_resolved.json"
    resolved_manifest = {
        "manifest_path": str(manifest_path),
        "filters": {
            "frameworks": _parse_cli_str_list(args.frameworks),
            "algorithms": _parse_cli_str_list(args.algorithms),
            "problem_ids": _parse_cli_int_list(args.problem_ids),
            "objective_counts": _parse_cli_int_list(args.objective_counts),
            "scenarios": _parse_cli_str_list(args.scenarios),
            "seeds": _parse_cli_int_list(args.seeds),
            "max_runs": int(args.max_runs or 0),
            "workers": workers,
        },
        "deadline": {
            "stop_new_runs_at": deadline.strftime("%Y-%m-%d %H:%M:%S") if deadline is not None else None,
            "stopped_by_deadline": stopped_by_deadline,
        },
    }
    resolved_manifest_path.write_text(json.dumps(resolved_manifest, indent=2), encoding="utf-8")

    ok_count = sum(1 for r in new_rows if r.get("status") == "ok")
    fail_count = sum(1 for r in new_rows if r.get("status") != "ok")
    skipped_by_deadline = max(0, len(tasks) - len(new_rows))
    _print(
        f"run done: ok={ok_count}, fail={fail_count}, "
        f"skipped_by_deadline={skipped_by_deadline if stopped_by_deadline else 0}, wrote={runs_csv}"
    )
    return 0 if fail_count == 0 else 1


def command_analyze(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest, paths = _manifest_and_paths(manifest_path)
    metrics_cfg = manifest.get("metrics", {})
    ref_dir = paths["reference_front_dir"]
    output_dir = paths["output_dir"]
    analysis_dir = output_dir / "analysis"
    _ensure_dir(analysis_dir)

    runs_csv = Path(args.runs_csv).resolve() if args.runs_csv else output_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    run_rows = _load_rows_csv(runs_csv)
    ok_rows = [r for r in run_rows if r.get("status") == "ok" and r.get("front_path")]
    if not ok_rows:
        raise RuntimeError("No successful runs found in runs.csv.")

    hv_padding = float(metrics_cfg.get("hv_ref_point_padding", 0.1))

    metric_rows: list[dict[str, Any]] = []
    ref_cache: dict[tuple[int, int], np.ndarray] = {}
    ref_point_cache: dict[tuple[int, int], np.ndarray] = {}

    for row in ok_rows:
        problem_id = int(row["problem_id"])
        n_obj = int(row["n_obj"])
        key = (problem_id, n_obj)

        if key not in ref_cache:
            ref_cache[key] = _read_reference_front(ref_dir, problem_id, n_obj)
            ref = ref_cache[key]
            min_ref = np.min(ref, axis=0)
            max_ref = np.max(ref, axis=0)
            span = np.maximum(max_ref - min_ref, 1.0e-12)
            ref_point_cache[key] = max_ref + hv_padding * span

        ref = ref_cache[key]
        ref_point = ref_point_cache[key]
        front = _read_matrix(Path(row["front_path"]))
        front = _nondominated(front)

        metrics = {
            "igd_plus": _igd_plus(front, ref),
            "epsilon_additive": _epsilon_additive(front, ref),
            "hv_normalized": _hv_normalized(front, ref, ref_point),
        }

        metric_rows.append(
            {
                "framework": row["framework"],
                "algorithm": row["algorithm"],
                "scenario_id": row["scenario_id"],
                "problem_id": problem_id,
                "n_obj": n_obj,
                "seed": int(row["seed"]),
                "runtime_seconds": float(row.get("runtime_seconds", "nan")),
                **metrics,
            }
        )

    metrics_path = analysis_dir / "metrics.csv"
    _write_rows_csv(metrics_path, metric_rows)

    summary_rows: list[dict[str, Any]] = []
    group_keys = ("algorithm", "scenario_id", "n_obj", "problem_id", "framework")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    indicator_names = [str(v) for v in metrics_cfg.get("indicators", ["igd_plus", "epsilon_additive", "hv_normalized"])]
    for key, rows in grouped.items():
        base = {k: v for k, v in zip(group_keys, key, strict=True)}
        for metric in indicator_names:
            values = np.asarray([float(r[metric]) for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            summary_rows.append(
                {
                    **base,
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )

    summary_path = analysis_dir / "summary.csv"
    _write_rows_csv(summary_path, summary_rows)

    pairwise_rows: list[dict[str, Any]] = []
    friedman_rows: list[dict[str, Any]] = []
    group_for_tests = ("algorithm", "scenario_id", "n_obj", "problem_id")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = tuple(row[k] for k in group_for_tests)
        groups.setdefault(key, []).append(row)

    for key, rows in groups.items():
        base = {k: v for k, v in zip(group_for_tests, key, strict=True)}
        for metric in indicator_names:
            metric_rows_local = [r for r in rows if math.isfinite(float(r[metric]))]
            if not metric_rows_local:
                continue

            by_fw_seed: dict[str, dict[int, float]] = {}
            for r in metric_rows_local:
                fw = str(r["framework"])
                by_fw_seed.setdefault(fw, {})
                by_fw_seed[fw][int(r["seed"])] = float(r[metric])

            frameworks = sorted(by_fw_seed)
            if len(frameworks) < 2:
                continue

            raw_pvals: list[float] = []
            pw_indices: list[int] = []
            for fw_a, fw_b in combinations(frameworks, 2):
                shared = sorted(set(by_fw_seed[fw_a]).intersection(by_fw_seed[fw_b]))
                if len(shared) < 2:
                    continue
                x = np.asarray([by_fw_seed[fw_a][s] for s in shared], dtype=float)
                y = np.asarray([by_fw_seed[fw_b][s] for s in shared], dtype=float)
                if np.allclose(x, y, equal_nan=False):
                    p_value = 1.0
                    statistic = 0.0
                else:
                    try:
                        stat = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="auto")
                        p_value = float(stat.pvalue)
                        statistic = float(stat.statistic)
                    except ValueError:
                        p_value = 1.0
                        statistic = float("nan")
                delta = _cliffs_delta(x, y)
                pw_row = {
                    **base,
                    "metric": metric,
                    "framework_a": fw_a,
                    "framework_b": fw_b,
                    "n_pairs": int(len(shared)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                    "cliffs_delta": delta,
                    "direction": METRIC_DIRECTIONS.get(metric, "min"),
                }
                pw_indices.append(len(pairwise_rows))
                pairwise_rows.append(pw_row)
                raw_pvals.append(p_value)

            if raw_pvals:
                adjusted = _holm_adjust(raw_pvals)
                for i, p_adj in zip(pw_indices, adjusted, strict=True):
                    pairwise_rows[i]["wilcoxon_p_holm"] = p_adj
                    pairwise_rows[i]["reject_0_05"] = bool(p_adj < 0.05)

            if len(frameworks) >= 3:
                shared_all = sorted(set.intersection(*[set(by_fw_seed[fw]) for fw in frameworks]))
                if len(shared_all) >= 2:
                    samples = [np.asarray([by_fw_seed[fw][s] for s in shared_all], dtype=float) for fw in frameworks]
                    if all(np.allclose(samples[0], s, equal_nan=False) for s in samples[1:]):
                        friedman_rows.append(
                            {
                                **base,
                                "metric": metric,
                                "n_frameworks": len(frameworks),
                                "n_blocks": len(shared_all),
                                "friedman_statistic": 0.0,
                                "friedman_p": 1.0,
                            }
                        )
                    else:
                        try:
                            fr = friedmanchisquare(*samples)
                            friedman_rows.append(
                                {
                                    **base,
                                    "metric": metric,
                                    "n_frameworks": len(frameworks),
                                    "n_blocks": len(shared_all),
                                    "friedman_statistic": float(fr.statistic),
                                    "friedman_p": float(fr.pvalue),
                                }
                            )
                        except Exception:
                            pass

    pairwise_path = analysis_dir / "pairwise_wilcoxon.csv"
    friedman_path = analysis_dir / "friedman.csv"
    _write_rows_csv(pairwise_path, pairwise_rows)
    _write_rows_csv(friedman_path, friedman_rows)

    _print(f"analyze done: metrics={len(metric_rows)}, summary={len(summary_rows)}, pairwise={len(pairwise_rows)}, friedman={len(friedman_rows)}")
    _print(f"wrote: {metrics_path}")
    _print(f"wrote: {summary_path}")
    _print(f"wrote: {pairwise_path}")
    _print(f"wrote: {friedman_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-framework ZCAT experiment runner (VAMOS vs jMetalPy vs pymoo).")
    parser.add_argument(
        "--manifest",
        default="experiments/configs/zcat_cross_framework.yaml",
        help="Path to YAML manifest.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Run objective/reference-front equivalence checks.")
    p_validate.add_argument("--problem-ids", default=None, help="Comma-separated subset override.")
    p_validate.add_argument("--objective-counts", default=None, help="Comma-separated subset override.")
    p_validate.add_argument("--n-points", type=int, default=None, help="Random points per configuration.")
    p_validate.add_argument("--tolerance", type=float, default=None, help="Numeric tolerance.")
    p_validate.add_argument("--seed", type=int, default=2026, help="RNG seed for random decision points.")
    p_validate.set_defaults(func=command_validate)

    p_run = sub.add_parser("run", help="Execute campaign runs.")
    p_run.add_argument("--frameworks", default=None, help="Comma-separated subset (vamos,jmetalpy,pymoo).")
    p_run.add_argument("--algorithms", default=None, help="Comma-separated subset (nsgaii,moead,smsemoa).")
    p_run.add_argument("--problem-ids", default=None, help="Comma-separated subset override.")
    p_run.add_argument("--objective-counts", default=None, help="Comma-separated subset override.")
    p_run.add_argument("--scenarios", default=None, help="Comma-separated subset override (e.g. S0,S1).")
    p_run.add_argument("--seeds", default=None, help="Comma-separated seed override (e.g. 1,2,3).")
    p_run.add_argument("--max-runs", type=int, default=0, help="Hard cap on number of tasks (0 means all).")
    p_run.add_argument("--workers", type=int, default=None, help="Worker processes. Use -1 for cpu_count()-1.")
    p_run.add_argument("--resume", action="store_true", help="Reuse existing run metadata when available.")
    p_run.add_argument("--debug", action="store_true", help="Print stack traces for failed runs.")
    p_run.set_defaults(func=command_run)

    p_analyze = sub.add_parser("analyze", help="Compute indicators and statistics from runs.csv.")
    p_analyze.add_argument("--runs-csv", default=None, help="Optional explicit runs.csv path.")
    p_analyze.set_defaults(func=command_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:  # pragma: no cover
        _print(f"fatal: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

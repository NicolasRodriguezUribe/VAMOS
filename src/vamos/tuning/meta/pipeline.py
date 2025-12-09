from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable, List, Sequence

import numpy as np

from vamos.algorithm.config import MOEADConfigData, NSGAIIConfigData, NSGAIIIConfigData, SMSEMOAConfigData
from vamos.study.runner import StudyRunner, StudyTask
from vamos.tuning.parameter_space import AlgorithmConfigSpace

from .nsga2_meta import MetaNSGAII
from .tuner import NSGAIITuner


def compute_hyperparameter_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    """Return absolute rank-correlations per dimension."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional.")
    if y.size != X.shape[0]:
        raise ValueError("y length must match number of rows in X.")
    names = names or [f"x{i}" for i in range(X.shape[1])]
    ranks_y = _rank_array(y)
    scores: list[tuple[str, float]] = []
    for idx in range(X.shape[1]):
        col = X[:, idx]
        ranks_x = _rank_array(col)
        corr = _safe_corr(ranks_x, ranks_y)
        scores.append((names[idx], abs(corr)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def _rank_array(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    cov = np.corrcoef(a, b)
    if cov.shape == (2, 2) and np.isfinite(cov[0, 1]):
        return float(cov[0, 1])
    return 0.0


class TuningPipeline:
    def __init__(
        self,
        problems: Sequence[Any],
        base_algorithm: str,
        config_space: AlgorithmConfigSpace,
        ref_fronts: Sequence[np.ndarray | None],
        indicators: Sequence[str],
        tuning_budget: dict,
        engine: str = "numpy",
        seed: int | None = None,
        *,
        tuner_factory: Callable[..., NSGAIITuner] = NSGAIITuner,
        study_runner_cls: Callable[..., StudyRunner] = StudyRunner,
        optimize_fn=None,
    ):
        self.problems = problems
        self.base_algorithm = base_algorithm
        self.config_space = config_space
        self.ref_fronts = ref_fronts
        self.indicators = indicators
        self.tuning_budget = dict(tuning_budget)
        self.engine = engine
        self.seed = seed
        self.tuner_factory = tuner_factory
        self.study_runner_cls = study_runner_cls
        self.optimize_fn = optimize_fn
        self._tuning_output: tuple[np.ndarray, np.ndarray, list[Any], dict] | None = None

    def run_tuning(self) -> None:
        budget = self._make_budget_defaults()
        tuner = self.tuner_factory(
            self.config_space,
            self.problems,
            self.ref_fronts,
            self.indicators,
            budget["max_evals_per_problem"],
            budget["n_runs_per_problem"],
            engine=self.engine,
            meta_population_size=budget["meta_population_size"],
            meta_max_evals=budget["meta_max_evals"],
            max_total_inner_runs=budget.get("max_total_inner_runs"),
            max_wall_time=budget.get("max_wall_time"),
            seed=self.seed,
            optimize_fn=self.optimize_fn,
            use_racing=budget.get("use_racing", False),
            baseline_quality=budget.get("baseline_quality"),
            min_runs_per_problem=budget.get("min_runs_per_problem"),
            max_runs_per_problem=budget.get("max_runs_per_problem"),
        )
        self._tuning_output = tuner.optimize()

    def select_top_k(self, k: int) -> list[Any]:
        if self._tuning_output is None:
            raise RuntimeError("run_tuning() must be called before selecting configurations.")
        _, F, configs, _ = self._tuning_output
        k = max(1, int(k))
        quality = F[:, 0]
        order = np.argsort(quality)
        return [configs[idx] for idx in order[:k]]

    def run_study(self, k: int, n_runs: int):
        configs = self.select_top_k(k)
        tasks: List[StudyTask] = []
        rng = np.random.default_rng(self.seed)
        for cfg in configs:
            algo_name = self._infer_algorithm_name(cfg) or self.base_algorithm
            for _ in range(n_runs):
                seed = int(rng.integers(0, 2**32 - 1))
                for problem in self.problems:
                    tasks.append(
                        StudyTask(
                            algorithm=algo_name,
                            engine=self.engine,
                            problem=self._problem_key(problem),
                            seed=seed,
                        )
                    )
        runner = self.study_runner_cls(verbose=False)
        return runner.run(tasks)

    def meta_dataframe(self):
        if self._tuning_output is None:
            raise RuntimeError("No tuning data available; call run_tuning() first.")
        X, F, configs, diagnostics = self._tuning_output
        names = self.config_space.parameter_names()
        data = []
        for idx in range(X.shape[0]):
            row = {names[i] if i < len(names) else f"x{i}": X[idx, i] for i in range(X.shape[1])}
            row.update(
                {
                    "obj_quality": F[idx, 0],
                    "obj_time": F[idx, 1],
                    "obj_robustness": F[idx, 2],
                    "config": configs[idx],
                }
            )
            data.append(row)
        try:
            import pandas as pd
        except Exception:  # pragma: no cover - optional dependency
            return data
        return pd.DataFrame(data), diagnostics

    def analyze_importance(self, objective_index: int = 0) -> list[tuple[str, float]]:
        if self._tuning_output is None:
            raise RuntimeError("No tuning data available; call run_tuning() first.")
        X, F, _, _ = self._tuning_output
        names = self.config_space.parameter_names()
        objective = F[:, objective_index]
        return compute_hyperparameter_importance(X, objective, names=names)

    def _make_budget_defaults(self) -> dict:
        return {
            "meta_population_size": self.tuning_budget.get("meta_population_size", 30),
            "meta_max_evals": self.tuning_budget.get("meta_max_evals", 100),
            "max_evals_per_problem": self.tuning_budget.get("max_evals_per_problem", 50),
            "n_runs_per_problem": self.tuning_budget.get("n_runs_per_problem", 1),
            "max_total_inner_runs": self.tuning_budget.get("max_total_inner_runs"),
            "max_wall_time": self.tuning_budget.get("max_wall_time"),
            "use_racing": self.tuning_budget.get("use_racing", False),
            "baseline_quality": self.tuning_budget.get("baseline_quality"),
            "min_runs_per_problem": self.tuning_budget.get("min_runs_per_problem"),
            "max_runs_per_problem": self.tuning_budget.get("max_runs_per_problem"),
        }

    @staticmethod
    def _infer_algorithm_name(config: Any) -> str | None:
        if isinstance(config, NSGAIIConfigData):
            return "nsgaii"
        if isinstance(config, MOEADConfigData):
            return "moead"
        if isinstance(config, SMSEMOAConfigData):
            return "smsemoa"
        if isinstance(config, NSGAIIIConfigData):
            return "nsgaiii"
        return None

    @staticmethod
    def _problem_key(problem: Any) -> str:
        if isinstance(problem, str):
            return problem
        return getattr(problem, "name", getattr(problem, "__class__", type("dummy", (), {})).__name__)

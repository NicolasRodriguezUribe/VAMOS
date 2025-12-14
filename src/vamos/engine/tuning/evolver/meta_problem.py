from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from vamos.engine.algorithm.components.hypervolume import hypervolume
from vamos.foundation.core.optimize import OptimizeConfig
from vamos.ux.analysis.core_objective_reduction import ObjectiveReductionConfig, reduce_objectives
from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace


class MetaOptimizationProblem:
    """
    Wraps inner algorithm executions as a single-objective minimization problem
    over the unit hypercube.
    """

    def __init__(
        self,
        config_space: AlgorithmConfigSpace,
        problems: Sequence[Any],
        ref_fronts: Sequence[np.ndarray | None],
        indicators: Sequence[str],
        max_evals_per_problem: int,
        n_runs_per_problem: int,
        engine: str,
        meta_rng: np.random.Generator,
        optimize_fn: Callable[..., Any] | None = None,
        objective_reduction: ObjectiveReductionConfig | None = None,
        *,
        min_runs_per_problem: int | None = None,
        max_runs_per_problem: int | None = None,
        use_racing: bool = False,
        baseline_quality: float | None = None,
    ):
        if not problems:
            raise ValueError("At least one problem is required for meta-optimization.")
        if len(problems) != len(ref_fronts):
            raise ValueError("ref_fronts must align one-to-one with problems.")
        self.config_space = config_space
        self.problems = list(problems)
        self.ref_fronts = [np.asarray(front) if front is not None else None for front in ref_fronts]
        self.indicators = [ind.lower() for ind in indicators]
        self.max_evals_per_problem = int(max_evals_per_problem)
        if self.max_evals_per_problem <= 0:
            raise ValueError("max_evals_per_problem must be positive.")
        self.n_runs_per_problem = int(n_runs_per_problem)
        if self.n_runs_per_problem <= 0:
            raise ValueError("n_runs_per_problem must be positive.")
        self.engine = engine
        self.meta_rng = meta_rng
        self._optimize = optimize_fn or self._import_optimize()
        self.objective_reduction = objective_reduction
        self.min_runs_per_problem = int(min_runs_per_problem or 1)
        if self.min_runs_per_problem <= 0:
            raise ValueError("min_runs_per_problem must be positive.")
        self.max_runs_per_problem = int(max_runs_per_problem) if max_runs_per_problem is not None else None
        if self.max_runs_per_problem is not None and self.max_runs_per_problem < self.min_runs_per_problem:
            raise ValueError("max_runs_per_problem must be >= min_runs_per_problem.")
        self.use_racing = bool(use_racing)
        self.baseline_quality = baseline_quality
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_inner_runs = 0
        self.last_elapsed_time = 0.0
        self.last_from_cache = False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self.last_inner_runs = 0
        self.last_elapsed_time = 0.0
        config = self.config_space.decode_vector(x)
        cache_key = self._make_cache_key(config)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self.last_from_cache = True
            return cached.copy()
        self.cache_misses += 1
        self.last_from_cache = False

        per_problem_quality: List[float] = []
        per_problem_time: List[float] = []
        per_problem_robust: List[float] = []
        termination = ("n_eval", self.max_evals_per_problem)
        algo = self._infer_algorithm_id(config)

        for p_idx, problem in enumerate(self.problems):
            run_scores: List[float] = []
            run_times: List[float] = []
            max_runs = self.max_runs_per_problem or self.n_runs_per_problem
            target_runs = self.n_runs_per_problem if not self.use_racing else max_runs
            for run_idx in range(target_runs):
                run_seed = int(self.meta_rng.integers(0, 2**32 - 1))
                start_run = time.perf_counter()
                result = self._optimize(
                    OptimizeConfig(
                        problem=problem,
                        algorithm=algo,
                        algorithm_config=config,
                        termination=termination,
                        seed=run_seed,
                        engine=self.engine,
                    )
                )
                elapsed = time.perf_counter() - start_run
                self.last_inner_runs += 1
                self.last_elapsed_time += elapsed
                F = self._extract_front(result)
                if F.ndim == 1:
                    F = F.reshape(-1, 1)
                nd_F = self._filter_nondominated(F)
                ref = self.ref_fronts[p_idx]
                if self.objective_reduction is not None:
                    target_dim = self.objective_reduction.target_dim
                    if target_dim is None:
                        raise ValueError(
                            "objective_reduction.target_dim must be set when objective reduction is enabled."
                        )
                    if nd_F.shape[1] > target_dim:
                        nd_F, selected = reduce_objectives(
                            nd_F,
                            method=self.objective_reduction.method,
                            target_dim=target_dim,
                            corr_threshold=self.objective_reduction.corr_threshold,
                            angle_threshold_deg=self.objective_reduction.angle_threshold_deg,
                            keep_mandatory=self.objective_reduction.keep_mandatory,
                        )
                        if ref is not None:
                            ref = ref[:, selected]
                quality = self._scalar_quality(nd_F, p_idx)
                run_scores.append(quality)
                run_times.append(elapsed)

                if self.use_racing and len(run_scores) >= self.min_runs_per_problem:
                    median_quality = float(np.median(run_scores))
                    if self.baseline_quality is not None and median_quality < self.baseline_quality:
                        break
            if not run_scores:
                raise RuntimeError("Meta-evaluation produced no runs for a problem.")
            per_problem_quality.append(float(np.median(run_scores)))
            per_problem_time.append(float(np.mean(run_times)))
            per_problem_robust.append(float(np.std(run_scores)))

        quality_obj = -float(np.mean(per_problem_quality))
        time_obj = float(np.mean(per_problem_time))
        robust_obj = float(np.mean(per_problem_robust))
        objectives = np.asarray([quality_obj, time_obj, robust_obj], dtype=float)
        self.cache[cache_key] = objectives
        return objectives

    def cache_key_for_vector(self, x: np.ndarray) -> str:
        """Return the cache key for a meta-vector without performing a full evaluate."""
        config = self.config_space.decode_vector(x)
        return self._make_cache_key(config)

    def cached_objectives(self, cache_key: str):
        """Return cached objectives for a cache key if present."""
        cached = self.cache.get(cache_key)
        return None if cached is None else cached.copy()

    def _scalar_quality(self, F: np.ndarray, problem_idx: int) -> float:
        if F.size == 0:
            return float("-inf")
        use_hv = "hv" in self.indicators
        if use_hv:
            ref_point = self._reference_point(problem_idx, F)
            return float(hypervolume(F, ref_point))
        return -float(np.mean(F[:, 0]))

    def _reference_point(self, problem_idx: int, F: np.ndarray) -> np.ndarray:
        ref_front = self.ref_fronts[problem_idx]
        base = np.atleast_2d(ref_front if ref_front is not None else F)
        return base.max(axis=0) + 1.0

    @staticmethod
    def _extract_front(result: Any) -> np.ndarray:
        if hasattr(result, "F"):
            return np.asarray(result.F, dtype=float)
        if isinstance(result, dict) and "F" in result:
            return np.asarray(result["F"], dtype=float)
        raise AttributeError("Optimization result must expose an objective matrix 'F'.")

    @staticmethod
    def _filter_nondominated(F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        if F.ndim != 2:
            raise ValueError("Objective matrix must be 2-dimensional.")
        n_points = F.shape[0]
        if n_points <= 1:
            return F
        keep = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            if not keep[i]:
                continue
            for j in range(n_points):
                if i == j or not keep[j]:
                    continue
                dominates = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                if dominates:
                    keep[i] = False
                    break
        return F[keep]

    def _make_cache_key(self, config: Any) -> str:
        cfg_dict = self._config_to_dict(config)
        payload = {
            "config": cfg_dict,
            "problems": tuple(self._problem_name(p) for p in self.problems),
            "max_eval": self.max_evals_per_problem,
            "runs": self.n_runs_per_problem,
            "min_runs": self.min_runs_per_problem,
            "max_runs": self.max_runs_per_problem,
            "indicators": tuple(self.indicators),
            "use_racing": self.use_racing,
            "baseline": self.baseline_quality,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _config_to_dict(config: Any) -> Dict[str, Any]:
        if hasattr(config, "to_dict"):
            return config.to_dict()
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        raise TypeError("Unsupported config type for caching.")

    @staticmethod
    def _problem_name(problem: Any) -> str:
        return getattr(problem, "name", problem.__class__.__name__)

    def get_cache_stats(self) -> dict:
        return {"hits": self.cache_hits, "misses": self.cache_misses}

    @staticmethod
    def _infer_algorithm_id(config: Any) -> str:
        by_type = {
            "NSGAIIConfigData": "nsgaii",
            "NSGAIIIConfigData": "nsga3",
            "MOEADConfigData": "moead",
            "SMSEMOAConfigData": "smsemoa",
            "SPEA2ConfigData": "spea2",
            "IBEAConfigData": "ibea",
            "SMPSOConfigData": "smpso",
        }
        algo = by_type.get(type(config).__name__)
        if algo:
            return algo
        if isinstance(config, dict):
            algo_dict = str(config.get("algorithm", "")).lower()
            if algo_dict:
                return algo_dict
        raise ValueError(
            "Unable to infer inner algorithm id from config; "
            "expected an engine.algorithm.config.*ConfigData instance."
        )

    @staticmethod
    def _import_optimize() -> Callable[[OptimizeConfig], Any]:
        try:
            from vamos.foundation.core.optimize import optimize
        except Exception as exc:
            raise ImportError("MetaOptimizationProblem requires 'vamos.foundation.core.optimize'.") from exc
        return optimize

from __future__ import annotations

import importlib.util
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Sequence

import numpy as np

from .racing.param_space import Boolean, Categorical, Int, ParamSpace, Real
from .racing.random_search_tuner import TrialResult
from .racing.tuning_task import EvalContext, Instance, TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def available_model_based_backends() -> dict[str, bool]:
    """Return availability flags for optional model-based tuning backends."""
    has_optuna = importlib.util.find_spec("optuna") is not None
    has_smac = importlib.util.find_spec("smac") is not None
    has_cs = importlib.util.find_spec("ConfigSpace") is not None
    has_hpbandster = importlib.util.find_spec("hpbandster") is not None
    return {
        "optuna": has_optuna,
        "bohb_optuna": has_optuna,
        "smac3": has_smac and has_cs,
        "bohb": has_hpbandster and has_cs,
    }


def _require_backend(name: str) -> None:
    available = available_model_based_backends()
    ok = bool(available.get(name, False))
    if ok:
        return
    if name == "optuna":
        raise RuntimeError("Backend 'optuna' requires optional dependency `optuna`.")
    if name == "bohb_optuna":
        raise RuntimeError("Backend 'bohb_optuna' requires optional dependency `optuna`.")
    if name == "smac3":
        raise RuntimeError("Backend 'smac3' requires optional dependencies `smac` and `ConfigSpace`.")
    if name == "bohb":
        raise RuntimeError("Backend 'bohb' requires optional dependencies `hpbandster` and `ConfigSpace`.")
    raise ValueError(f"Unknown backend '{name}'.")


def _sample_from_optuna_trial(trial: Any, param_space: ParamSpace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for name, spec in param_space.params.items():
        if isinstance(spec, Real):
            cfg[name] = float(trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Int):
            cfg[name] = int(trial.suggest_int(name, int(spec.low), int(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Categorical):
            cfg[name] = trial.suggest_categorical(name, list(spec.choices))
        elif isinstance(spec, Boolean):
            cfg[name] = bool(trial.suggest_categorical(name, [False, True]))
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
    return cfg


def _build_configspace(param_space: ParamSpace, seed: int):
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    cs = ConfigurationSpace(seed=int(seed))
    for name, spec in param_space.params.items():
        if isinstance(spec, Real):
            hp = UniformFloatHyperparameter(name=name, lower=float(spec.low), upper=float(spec.high), log=bool(spec.log))
        elif isinstance(spec, Int):
            hp = UniformIntegerHyperparameter(name=name, lower=int(spec.low), upper=int(spec.high), log=bool(spec.log))
        elif isinstance(spec, Categorical):
            hp = CategoricalHyperparameter(name=name, choices=list(spec.choices))
        elif isinstance(spec, Boolean):
            hp = CategoricalHyperparameter(name=name, choices=[False, True])
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
        cs.add_hyperparameter(hp)
    return cs


def _estimate_hyperband_evals_per_iteration(max_budget: int, eta: int) -> int:
    max_budget = max(1, int(max_budget))
    eta = max(2, int(eta))
    if max_budget <= 1:
        return 1
    min_budget = 1.0
    s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))
    B = (s_max + 1) * max_budget
    total = 0
    for s in range(s_max, -1, -1):
        n = int(np.ceil((B / max_budget / (s + 1)) * (eta**s)))
        n = max(1, n)
        for i in range(s + 1):
            n_i = int(np.floor(n * (eta**(-i))))
            total += max(1, n_i)
    return max(1, total)


def _suite_key(instance: Instance) -> str:
    kwargs = dict(getattr(instance, "kwargs", {}) or {})
    for key in ("suite", "family", "group"):
        value = kwargs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    name = str(getattr(instance, "name", "")).strip().lower()
    if "_" in name:
        return name.split("_", 1)[0]
    return name if name else "default"


@dataclass
class ModelBasedTuner:
    """
    Optional model-based tuning facade for external backends.

    Backends:
    - `optuna`: TPE + MedianPruner
    - `bohb_optuna`: TPE + HyperbandPruner
    - `smac3`: SMAC3 MultiFidelityFacade
    - `bohb`: hpbandster BOHB
    """

    task: TuningTask
    max_trials: int
    backend: str = "optuna"
    seed: int = 0
    n_jobs: int = 1
    timeout_seconds: float | None = None
    show_progress_bar: bool = False
    bohb_reduction_factor: int = 3
    budget_levels: list[int] | None = None
    fidelity_min_instance_frac: float = 1.0
    fidelity_min_seed_count: int | None = None
    fidelity_max_seed_count: int | None = None
    fidelity_selection_seed: int | None = None
    optuna_storage_url: str | None = None
    optuna_study_name: str | None = None
    optuna_load_if_exists: bool = True
    _fidelity_cache: dict[int, tuple[Sequence[Instance], Sequence[int], dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        frac = float(self.fidelity_min_instance_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("fidelity_min_instance_frac must be in (0, 1].")
        if self.fidelity_min_seed_count is not None and int(self.fidelity_min_seed_count) <= 0:
            raise ValueError("fidelity_min_seed_count must be > 0 when provided.")
        if self.fidelity_max_seed_count is not None and int(self.fidelity_max_seed_count) <= 0:
            raise ValueError("fidelity_max_seed_count must be > 0 when provided.")
        if (
            self.fidelity_min_seed_count is not None
            and self.fidelity_max_seed_count is not None
            and int(self.fidelity_min_seed_count) > int(self.fidelity_max_seed_count)
        ):
            raise ValueError("fidelity_min_seed_count cannot be greater than fidelity_max_seed_count.")

    def _worst_score(self) -> float:
        return float("-inf") if self.task.maximize else float("inf")

    def _score_to_loss(self, score: float) -> float:
        return float(-score if self.task.maximize else score)

    def _default_optuna_study_name(self) -> str:
        raw = f"{self.task.name}_{self.backend}_{int(self.seed)}"
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)
        return safe or f"vamos_tuning_{int(self.seed)}"

    def _resolve_seed_bounds(self) -> tuple[int, int]:
        total = max(1, len(self.task.seeds))
        max_count = int(self.fidelity_max_seed_count) if self.fidelity_max_seed_count is not None else total
        max_count = min(total, max(1, max_count))
        min_count = int(self.fidelity_min_seed_count) if self.fidelity_min_seed_count is not None else max_count
        min_count = min(max_count, max(1, min_count))
        return min_count, max_count

    def _budget_fraction(self, budget: int) -> float:
        b = int(min(int(self.task.budget_per_run), max(1, int(budget))))
        if self.budget_levels:
            levels = self._resolve_budget_levels()
            b_min = int(levels[0])
            b_max = int(levels[-1])
        else:
            b_min = 1
            b_max = max(1, int(self.task.budget_per_run))
        if b_max <= b_min:
            return 1.0
        frac = (float(b) - float(b_min)) / float(b_max - b_min)
        return float(np.clip(frac, 0.0, 1.0))

    def _fidelity_level_info(self, budget: int) -> tuple[int, int | None]:
        levels = self._resolve_budget_levels()
        b = int(min(int(self.task.budget_per_run), max(1, int(budget))))
        idx = 0
        for i, level in enumerate(levels):
            if b >= int(level):
                idx = int(i)
            else:
                break
        prev = int(levels[idx - 1]) if idx > 0 else None
        return int(idx), prev

    def _resolve_fidelity_slice(self, budget: int) -> tuple[Sequence[Instance], Sequence[int], dict[str, Any]]:
        b = int(min(int(self.task.budget_per_run), max(1, int(budget))))
        cached = self._fidelity_cache.get(int(b))
        if cached is not None:
            return cached

        all_instances = list(self.task.instances)
        all_seeds = [int(s) for s in self.task.seeds]
        if not all_instances:
            raise RuntimeError("Tuning task has no instances.")
        if not all_seeds:
            raise RuntimeError("Tuning task has no seeds.")

        frac = self._budget_fraction(b)
        min_inst_frac = float(self.fidelity_min_instance_frac)
        inst_frac = min_inst_frac + frac * (1.0 - min_inst_frac)
        target_instances = int(max(1, min(len(all_instances), int(round(float(len(all_instances)) * inst_frac)))))
        if target_instances >= len(all_instances):
            selected_instances = list(all_instances)
        else:
            rng_seed = int(self.fidelity_selection_seed if self.fidelity_selection_seed is not None else self.seed)
            rng = np.random.default_rng(rng_seed + int(7919 * b))
            groups: dict[str, list[int]] = {}
            for idx, inst in enumerate(all_instances):
                groups.setdefault(_suite_key(inst), []).append(int(idx))
            group_names = sorted(groups)
            for name in group_names:
                rng.shuffle(groups[name])

            selected_idx: list[int] = []
            if target_instances <= len(group_names):
                chosen = rng.choice(np.asarray(group_names, dtype=object), size=target_instances, replace=False)
                for name in chosen.tolist():
                    selected_idx.append(int(groups[str(name)].pop()))
            else:
                for name in group_names:
                    selected_idx.append(int(groups[name].pop()))
                remainder: list[int] = []
                for name in group_names:
                    remainder.extend(int(v) for v in groups[name])
                rng.shuffle(remainder)
                missing = int(target_instances - len(selected_idx))
                selected_idx.extend(remainder[:missing])

            selected_instances = [all_instances[i] for i in sorted(set(selected_idx))]
            if len(selected_instances) < target_instances:
                used = {id(inst) for inst in selected_instances}
                for inst in all_instances:
                    if id(inst) in used:
                        continue
                    selected_instances.append(inst)
                    if len(selected_instances) >= target_instances:
                        break

        min_seed_count, max_seed_count = self._resolve_seed_bounds()
        target_seed_count = int(round(float(min_seed_count) + frac * float(max_seed_count - min_seed_count)))
        target_seed_count = max(1, min(len(all_seeds), target_seed_count))
        selected_seeds = list(all_seeds[:target_seed_count]) if target_seed_count < len(all_seeds) else list(all_seeds)

        fidelity_level, previous_budget = self._fidelity_level_info(b)
        meta = {
            "budget": int(b),
            "budget_fraction": float(frac),
            "instances_used": int(len(selected_instances)),
            "instances_total": int(len(all_instances)),
            "seeds_used": int(len(selected_seeds)),
            "seeds_total": int(len(all_seeds)),
            "fidelity_level": int(fidelity_level),
            "previous_budget": (None if previous_budget is None else int(previous_budget)),
        }
        resolved = (selected_instances, selected_seeds, meta)
        self._fidelity_cache[int(b)] = resolved
        return resolved

    def _eval_config_at_budget(
        self,
        config: dict[str, Any],
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        budget: int,
    ) -> float:
        scores: list[float] = []
        b = int(min(int(self.task.budget_per_run), max(1, int(budget))))
        instances, seeds, fidelity_meta = self._resolve_fidelity_slice(b)
        fidelity_level = int(fidelity_meta.get("fidelity_level", 0))
        previous_budget = fidelity_meta.get("previous_budget", None)
        for inst in instances:
            for seed in seeds:
                ctx = EvalContext(
                    instance=inst,
                    seed=int(seed),
                    budget=b,
                    fidelity_level=int(fidelity_level),
                    previous_budget=(None if previous_budget is None else int(previous_budget)),
                )
                result = eval_fn(config, ctx)
                if isinstance(result, tuple):
                    scores.append(float(result[0]))
                else:
                    scores.append(float(result))
        if not scores:
            raise RuntimeError("No scores computed for configuration.")
        return float(self.task.aggregator(scores))

    def _resolve_budget_levels(self) -> list[int]:
        if self.budget_levels:
            levels = [int(v) for v in self.budget_levels if int(v) > 0]
            if not levels:
                levels = [int(self.task.budget_per_run)]
            levels = sorted(set(min(int(self.task.budget_per_run), max(1, v)) for v in levels))
        else:
            bmax = max(1, int(self.task.budget_per_run))
            if bmax <= 3:
                levels = list(range(1, bmax + 1))
            else:
                levels = sorted(set([max(1, bmax // 3), max(1, (2 * bmax) // 3), bmax]))
        if levels[-1] != int(self.task.budget_per_run):
            levels.append(int(self.task.budget_per_run))
        return levels

    def _run_optuna_like(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        bohb_mode: bool,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        import optuna

        levels = self._resolve_budget_levels()
        sampler = optuna.samplers.TPESampler(
            seed=int(self.seed),
            multivariate=True,
            group=True,
        )
        if bohb_mode:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=max(1, len(levels)),
                reduction_factor=max(2, int(self.bohb_reduction_factor)),
            )
        else:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=max(5, min(40, int(self.max_trials // 4) if self.max_trials > 0 else 5)),
                n_warmup_steps=max(1, len(levels) - 1),
                interval_steps=1,
            )

        storage_url = str(self.optuna_storage_url or "").strip() or None
        study_name_raw = str(self.optuna_study_name or "").strip()
        study_name = study_name_raw if study_name_raw else self._default_optuna_study_name()
        create_kwargs: dict[str, Any] = {
            "direction": "maximize" if self.task.maximize else "minimize",
            "sampler": sampler,
            "pruner": pruner,
        }
        if storage_url:
            create_kwargs["storage"] = storage_url
            create_kwargs["study_name"] = study_name
            create_kwargs["load_if_exists"] = bool(self.optuna_load_if_exists)

        study = optuna.create_study(**create_kwargs)

        def objective(trial: Any) -> float:
            config = _sample_from_optuna_trial(trial, self.task.param_space)
            self.task.param_space.validate(config)
            trial.set_user_attr("config", dict(config))
            final_score = self._worst_score()
            fidelity_trace: list[dict[str, Any]] = []
            for step_idx, budget in enumerate(levels):
                score = self._eval_config_at_budget(config, eval_fn, budget=int(budget))
                final_score = score
                _, _, fidelity_meta = self._resolve_fidelity_slice(int(budget))
                fidelity_trace.append(
                    {
                        "step": int(step_idx),
                        "budget": int(budget),
                        "score": float(score),
                        "instances_used": int(fidelity_meta.get("instances_used", 0)),
                        "instances_total": int(fidelity_meta.get("instances_total", 0)),
                        "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
                        "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
                    }
                )
                trial.report(float(score), step=step_idx)
                if trial.should_prune():
                    trial.set_user_attr("fidelity_trace", fidelity_trace)
                    raise optuna.TrialPruned(f"Pruned at step {step_idx} (budget={budget}).")
            trial.set_user_attr("fidelity_trace", fidelity_trace)
            return float(final_score)

        study.optimize(
            objective,
            n_trials=int(self.max_trials),
            n_jobs=int(self.n_jobs),
            timeout=float(self.timeout_seconds) if self.timeout_seconds is not None else None,
            show_progress_bar=bool(self.show_progress_bar),
        )

        history: list[TrialResult] = []
        for trial in study.trials:
            if trial.value is None:
                continue
            cfg = trial.user_attrs.get("config", dict(trial.params))
            details: dict[str, Any] = {"state": str(getattr(trial.state, "name", trial.state))}
            fidelity_trace = trial.user_attrs.get("fidelity_trace")
            if isinstance(fidelity_trace, list):
                details["fidelity_trace"] = fidelity_trace
            history.append(TrialResult(trial_id=int(trial.number), config=dict(cfg), score=float(trial.value), details=details))
        if not history:
            raise RuntimeError("Tuner finished without a valid configuration.")
        best_trial = study.best_trial
        best_cfg = best_trial.user_attrs.get("config", dict(best_trial.params))
        return dict(best_cfg), history

    def _run_smac3(self, eval_fn: Callable[[dict[str, Any], EvalContext], float]) -> tuple[dict[str, Any], list[TrialResult]]:
        from smac import MultiFidelityFacade, Scenario

        cs = _build_configspace(self.task.param_space, seed=int(self.seed))
        levels = self._resolve_budget_levels()
        if self.budget_levels:
            min_budget_i = int(levels[0])
            max_budget_i = int(levels[-1])
        else:
            min_budget_i = 1
            max_budget_i = int(self.task.budget_per_run)
        min_budget_i = max(1, min_budget_i)
        max_budget_i = max(min_budget_i, max_budget_i)

        def target(config, seed: int = 0, budget: float | None = None) -> float:
            cfg = dict(config)
            self.task.param_space.validate(cfg)
            b = int(round(float(budget if budget is not None else max_budget_i)))
            b = min(int(max_budget_i), max(int(min_budget_i), b))
            score = self._eval_config_at_budget(cfg, eval_fn, budget=b)
            return float(self._score_to_loss(score))

        scenario = Scenario(
            configspace=cs,
            deterministic=False,
            n_trials=int(self.max_trials),
            min_budget=float(min_budget_i),
            max_budget=float(max_budget_i),
            n_workers=int(max(1, self.n_jobs)),
            seed=int(self.seed),
            walltime_limit=float(self.timeout_seconds) if self.timeout_seconds is not None else np.inf,
        )
        optimizer = MultiFidelityFacade(
            scenario=scenario,
            target_function=target,
            overwrite=True,
        )
        _ = optimizer.optimize()

        history: list[TrialResult] = []
        runhistory = optimizer.runhistory
        for trial_id, (trial_key, trial_value) in enumerate(runhistory.items()):
            cfg = runhistory.get_config(trial_key.config_id)
            if cfg is None:
                continue
            cfg_dict = dict(cfg)
            raw_cost = trial_value.cost
            if isinstance(raw_cost, list):
                if not raw_cost:
                    continue
                loss = float(raw_cost[0])
            else:
                loss = float(raw_cost)
            score = float(-loss if self.task.maximize else loss)
            budget = None if trial_key.budget is None else int(round(float(trial_key.budget)))
            fidelity_budget = int(max_budget_i if budget is None else budget)
            _, _, fidelity_meta = self._resolve_fidelity_slice(int(fidelity_budget))
            details = {
                "backend": "smac3",
                "seed": None if trial_key.seed is None else int(trial_key.seed),
                "budget": budget,
                "loss": float(loss),
                "status": str(getattr(trial_value.status, "name", trial_value.status)),
                "time": float(trial_value.time),
                "cpu_time": float(trial_value.cpu_time),
                "instances_used": int(fidelity_meta.get("instances_used", 0)),
                "instances_total": int(fidelity_meta.get("instances_total", 0)),
                "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
                "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
            }
            history.append(TrialResult(trial_id=int(trial_id), config=cfg_dict, score=score, details=details))

        if not history:
            raise RuntimeError("Tuner finished without a valid configuration.")
        best = max(history, key=lambda h: h.score) if self.task.maximize else min(history, key=lambda h: h.score)
        return dict(best.config), history

    def _run_bohb_native(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        import hpbandster.core.nameserver as hpns
        from hpbandster.core.worker import Worker
        from hpbandster.optimizers import BOHB

        cs = _build_configspace(self.task.param_space, seed=int(self.seed))
        eta = max(2, int(self.bohb_reduction_factor))
        levels = self._resolve_budget_levels()
        if self.budget_levels:
            min_budget_i = int(levels[0])
            max_budget_i = int(levels[-1])
        else:
            min_budget_i = 1
            max_budget_i = int(self.task.budget_per_run)
        min_budget_i = max(1, min_budget_i)
        max_budget_i = max(min_budget_i, max_budget_i)
        history: list[TrialResult] = []
        lock = threading.Lock()
        trial_counter = 0

        class _Worker(Worker):
            def compute(inner_self, config, budget, **kwargs):
                nonlocal trial_counter
                cfg = dict(config)
                try:
                    self.task.param_space.validate(cfg)
                except Exception as exc:
                    return {"loss": 1e9, "info": {"error": f"{type(exc).__name__}: {exc}"}}
                b = int(round(float(budget if budget is not None else max_budget_i)))
                b = min(int(max_budget_i), max(int(min_budget_i), b))
                score = self._eval_config_at_budget(cfg, eval_fn, budget=b)
                loss = self._score_to_loss(score)
                _, _, fidelity_meta = self._resolve_fidelity_slice(int(b))
                with lock:
                    tid = int(trial_counter)
                    trial_counter += 1
                    history.append(
                        TrialResult(
                            trial_id=tid,
                            config=dict(cfg),
                            score=float(score),
                            details={
                                "backend": "bohb",
                                "budget": int(b),
                                "loss": float(loss),
                                "instances_used": int(fidelity_meta.get("instances_used", 0)),
                                "instances_total": int(fidelity_meta.get("instances_total", 0)),
                                "seeds_used": int(fidelity_meta.get("seeds_used", 0)),
                                "seeds_total": int(fidelity_meta.get("seeds_total", 0)),
                            },
                        )
                    )
                return {"loss": float(loss), "info": {"score": float(score)}}

        run_id = f"vamos_bohb_{int(time.time())}_{int(self.seed)}"
        ns = None
        optimizer = None
        workers: list[Any] = []
        try:
            ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=0)
            ns_host, ns_port = ns.start()
            for worker_id in range(max(1, int(self.n_jobs))):
                worker = _Worker(
                    run_id=run_id,
                    host="127.0.0.1",
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    id=worker_id,
                )
                worker.run(background=True)
                workers.append(worker)

            optimizer = BOHB(
                configspace=cs,
                run_id=run_id,
                nameserver=ns_host,
                nameserver_port=ns_port,
                min_budget=float(min_budget_i),
                max_budget=float(max_budget_i),
                eta=int(eta),
                random_state=int(self.seed),
            )
            evals_per_iter = _estimate_hyperband_evals_per_iteration(max_budget=int(max_budget_i), eta=int(eta))
            n_iterations = max(1, int(math.ceil(max(1, int(self.max_trials)) / evals_per_iter)))
            _ = optimizer.run(n_iterations=n_iterations, min_n_workers=max(1, int(self.n_jobs)))
        finally:
            if optimizer is not None:
                try:
                    optimizer.shutdown(shutdown_workers=True)
                except Exception:
                    _logger().debug("Failed to shutdown BOHB optimizer cleanly.", exc_info=True)
            if ns is not None:
                try:
                    ns.shutdown()
                except Exception:
                    _logger().debug("Failed to shutdown BOHB nameserver cleanly.", exc_info=True)

        if not history:
            raise RuntimeError("Tuner finished without a valid configuration.")
        best = max(history, key=lambda h: h.score) if self.task.maximize else min(history, key=lambda h: h.score)
        return dict(best.config), history

    def run(
        self,
        eval_fn: Callable[[dict[str, Any], EvalContext], float],
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        _ = verbose
        if self.backend == "optuna":
            _require_backend("optuna")
            return self._run_optuna_like(eval_fn, bohb_mode=False)
        if self.backend == "bohb_optuna":
            _require_backend("bohb_optuna")
            return self._run_optuna_like(eval_fn, bohb_mode=True)
        if self.backend == "smac3":
            _require_backend("smac3")
            return self._run_smac3(eval_fn)
        if self.backend == "bohb":
            _require_backend("bohb")
            return self._run_bohb_native(eval_fn)
        raise ValueError(f"Unknown backend '{self.backend}'.")


__all__ = ["ModelBasedTuner", "available_model_based_backends"]

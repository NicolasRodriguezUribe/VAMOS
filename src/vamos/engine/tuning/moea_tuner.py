"""
Domain-specific MOEA tuner with hierarchical phase-based racing.

Unlike general-purpose tuners that treat algorithm configuration as black-box
optimization, the MOEATuner exploits MOEA-specific knowledge:

1. **Parameter semantics**: Parameters are tagged with roles (structural,
   operator, operator_rate, population, adaptive) and tuned hierarchically —
   structural choices first, then operator families, then fine-tuning rates.

2. **Convergence-aware evaluation**: Runs are monitored via per-generation HV
   callbacks. Convergence profiles enable adaptive early stopping, composite
   scoring, and diagnostic-driven sampling.

3. **Knowledge transfer**: A persistent knowledge base stores tuning results
   keyed by problem features, enabling warm-start sampling for new problems.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .convergence import ConvergenceDiagnostic, compute_profile, diagnose
from .knowledge_base import KnowledgeBase
from .problem_features import extract_features_from_instances
from .racing.config_space import AlgorithmConfigSpace
from .racing.core import RacingTuner
from .racing.eval_types import EvalFn
from .racing.param_space import Boolean, Categorical, Int, ParamSpace, ParamType, Real
from .racing.random_search_tuner import TrialResult
from .racing.scenario import Scenario
from .racing.tuning_task import EvalContext, TuningTask


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


# Phase definitions: role groups tuned in each phase
PHASE_A_ROLES = ("structural", "population")
PHASE_B_ROLES = ("operator",)
PHASE_C_ROLES = ("operator_rate", "adaptive")
_CONDITION_CFG_KEY_PATTERN = r"cfg\['([^']+)'\]"


@dataclass
class PhaseResult:
    """Result of a single hierarchical tuning phase."""

    phase: str
    best_config: dict[str, Any]
    history: list[TrialResult]
    diagnostics: list[ConvergenceDiagnostic] = field(default_factory=list)


@dataclass
class MOEATunerConfig:
    """Configuration for the MOEATuner.

    Parameters
    ----------
    max_experiments : int
        Total experiment budget across all phases.
    budget_split : tuple of 3 floats
        Fraction of max_experiments allocated to Phase A, B, C.
        Must sum to 1.0.
    budget_per_run_phases : tuple of 3 ints or None
        Evaluation budget (function evaluations) for each phase's runs.
        Phase A uses cheap runs, Phase C uses full budget.
        If None, uses (budget_per_run * 0.2, * 0.5, * 1.0).
    max_initial_configs : int
        Number of initial configurations to sample per phase.
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel evaluation jobs.
    use_statistical_tests : bool
        Whether to use Friedman + paired t-tests for elimination.
    verbose : bool
        Whether to print progress.
    """

    max_experiments: int = 200
    budget_split: tuple[float, float, float] = (0.3, 0.35, 0.35)
    budget_per_run_phases: tuple[int, int, int] | None = None
    max_initial_configs: int = 20
    seed: int = 0
    n_jobs: int = 1
    use_statistical_tests: bool = True
    verbose: bool = True
    use_knowledge_base: bool = False
    knowledge_base_path: str | None = None

    def __post_init__(self) -> None:
        total = sum(self.budget_split)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"budget_split must sum to 1.0, got {total}")
        if self.max_experiments <= 0:
            raise ValueError("max_experiments must be > 0")
        if self.max_initial_configs <= 0:
            raise ValueError("max_initial_configs must be > 0")


class MOEATuner:
    """Domain-specific MOEA tuner with hierarchical phase-based racing.

    The tuner exploits parameter role annotations to search the configuration
    space in three phases:

    - **Phase A (Structure + Population)**: Races over structural parameters
      (selection strategy, initializer, repair) and population parameters
      (pop_size, offspring_ratio). Uses cheap evaluation budgets to eliminate
      clearly bad structural choices fast.

    - **Phase B (Operators)**: Fixes the best structural/population choices
      from Phase A and races over operator families (crossover type, mutation
      type). Moderate evaluation budgets.

    - **Phase C (Fine-tuning)**: Fixes the best operators from Phase B and
      tunes continuous rates (eta, probabilities, sigmas) with full evaluation
      budgets. Smaller search space enables more precise tuning.

    Parameters
    ----------
    config_space : AlgorithmConfigSpace
        The full algorithm configuration space with role-tagged parameters.
    task : TuningTask
        Tuning task with instances, seeds, and evaluation budget.
    config : MOEATunerConfig
        Tuner configuration (budget split, parallelism, etc.).
    initial_configs : list of dict, optional
        Warm-start configurations (e.g., from a knowledge base).
    """

    def __init__(
        self,
        config_space: AlgorithmConfigSpace,
        task: TuningTask,
        config: MOEATunerConfig | None = None,
        initial_configs: list[dict[str, Any]] | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self.config_space = config_space
        self.task = task
        self.config = config or MOEATunerConfig()
        self.initial_configs = list(initial_configs or [])
        self.rng = np.random.default_rng(self.config.seed)

        self._phase_results: list[PhaseResult] = []

        # Knowledge base integration
        self._kb: KnowledgeBase | None = None
        if knowledge_base is not None:
            self._kb = knowledge_base
        elif self.config.use_knowledge_base:
            self._kb = KnowledgeBase(self.config.knowledge_base_path)

    def run(
        self,
        eval_fn: EvalFn,
        verbose: bool | None = None,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        """Run the hierarchical MOEA tuner.

        Parameters
        ----------
        eval_fn : callable
            Evaluation function: (config_dict, EvalContext) -> float | (float, Any)
        verbose : bool, optional
            Override verbosity.

        Returns
        -------
        best_config : dict
            Best configuration found.
        history : list of TrialResult
            Full trial history across all phases.
        """
        verbose_flag = self.config.verbose if verbose is None else verbose
        cfg = self.config

        # Query knowledge base for warm-start configs
        if self._kb is not None and not self.initial_configs:
            features = extract_features_from_instances(list(self.task.instances))
            algorithm = self.config_space.algorithm_name.split("_")[0]
            kb_configs = self._kb.get_warm_start_configs(features, algorithm=algorithm, k=5)
            if kb_configs:
                self.initial_configs.extend(kb_configs)
                if verbose_flag:
                    _logger().info("[moea-tuner] Warm-started with %d configs from knowledge base", len(kb_configs))

        # Compute per-phase budgets (exactly sums to max_experiments)
        phase_experiments = _allocate_phase_experiments(
            total=cfg.max_experiments,
            split=cfg.budget_split,
        )

        # Compute per-run budgets for each phase
        base_budget = self.task.budget_per_run or 50_000
        if cfg.budget_per_run_phases is not None:
            run_budgets = cfg.budget_per_run_phases
        else:
            run_budgets = (
                max(1, int(base_budget * 0.2)),
                max(1, int(base_budget * 0.5)),
                base_budget,
            )

        # Group parameters by role
        role_groups = self.config_space.group_by_role()
        full_param_space = self.config_space.to_param_space()

        frozen: dict[str, Any] = {}
        all_history: list[TrialResult] = []
        next_trial_id = 0
        phase_plan = [
            ("A (Structure + Population)", PHASE_A_ROLES, phase_experiments[0], run_budgets[0]),
            ("B (Operators)", PHASE_B_ROLES, phase_experiments[1], run_budgets[1]),
            ("C (Fine-tuning)", PHASE_C_ROLES, phase_experiments[2], run_budgets[2]),
        ]
        for phase_name, active_roles, phase_budget, run_budget in phase_plan:
            phase_result = self._run_phase(
                phase_name=phase_name,
                active_roles=active_roles,
                frozen_params=frozen,
                full_param_space=full_param_space,
                role_groups=role_groups,
                eval_fn=eval_fn,
                max_experiments=phase_budget,
                budget_per_run=run_budget,
                verbose=verbose_flag,
            )
            phase_result.history = _reindex_history(phase_result.history, next_trial_id)
            next_trial_id += len(phase_result.history)

            self._phase_results.append(phase_result)
            all_history.extend(phase_result.history)
            frozen.update(phase_result.best_config)
            self._seed_initial_configs_from_diagnostics(
                phase_result=phase_result,
                frozen_params=frozen,
                role_groups=role_groups,
            )
            if verbose_flag:
                _logger().info(
                    "[moea-tuner] Phase %s done. Frozen: %s",
                    phase_name,
                    _brief_config(phase_result.best_config),
                )

        if verbose_flag:
            _logger().info("[moea-tuner] All phases complete. Best config: %s", _brief_config(frozen))

        # Store results in knowledge base
        if self._kb is not None:
            features = extract_features_from_instances(list(self.task.instances))
            algorithm = self.config_space.algorithm_name.split("_")[0]
            encoding = self.config_space.algorithm_name.split("_", 1)[1] if "_" in self.config_space.algorithm_name else "real"
            best_score = max((t.score for t in all_history if t.score == t.score), default=0.0)  # noqa: PLR0124 (NaN check)
            self._kb.store(
                features=features,
                algorithm=algorithm,
                encoding=encoding,
                best_config=frozen,
                score=best_score,
                metadata={
                    "tuner": "moea_tuner",
                    "max_experiments": cfg.max_experiments,
                    "phases": len(self._phase_results),
                },
            )
            if verbose_flag:
                _logger().info("[moea-tuner] Stored result in knowledge base (score=%.4f)", best_score)

        return frozen, all_history

    def _run_phase(
        self,
        *,
        phase_name: str,
        active_roles: tuple[str, ...],
        frozen_params: dict[str, Any],
        full_param_space: ParamSpace,
        role_groups: dict[str, list[ParamType]],
        eval_fn: EvalFn,
        max_experiments: int,
        budget_per_run: int,
        verbose: bool,
    ) -> PhaseResult:
        """Run a single hierarchical phase, racing only over active-role params."""
        if max_experiments <= 0:
            if verbose:
                _logger().info("[moea-tuner] Skipping phase %s (budget=0)", phase_name)
            return PhaseResult(phase=phase_name, best_config={}, history=[], diagnostics=[])

        if verbose:
            _logger().info("[moea-tuner] Starting Phase %s (budget=%d experiments)", phase_name, max_experiments)

        # Build a reduced param space for this phase
        active_params: dict[str, ParamType] = {}
        for role in active_roles:
            for p in role_groups.get(role, []):
                active_params[p.name] = p

        if not active_params:
            if verbose:
                _logger().info("[moea-tuner] No active params in phase %s; skipping.", phase_name)
            return PhaseResult(phase=phase_name, best_config={}, history=[], diagnostics=[])

        active_conditions = [c for c in (full_param_space.conditions or []) if c.param_name in active_params]
        context_params = self._build_condition_context_params(
            conditions=active_conditions,
            active_params=active_params,
            frozen_params=frozen_params,
            full_param_space=full_param_space,
        )
        phase_space = ParamSpace(params={**active_params, **context_params}, conditions=active_conditions)

        # Create a wrapping eval_fn that merges frozen params with phase params
        def phase_eval_fn(config: dict[str, Any], ctx: EvalContext) -> float | tuple[float, Any]:
            merged = self._complete_config(
                {**frozen_params, **config},
                full_param_space,
            )
            return eval_fn(merged, ctx)

        # Create the task for this phase
        phase_task = TuningTask(
            name=f"{self.task.name}_phase_{phase_name}",
            param_space=phase_space,
            instances=list(self.task.instances),
            seeds=list(self.task.seeds),
            aggregator=self.task.aggregator,
            budget_per_run=budget_per_run,
            maximize=self.task.maximize,
        )

        scenario = Scenario(
            max_experiments=max_experiments,
            use_statistical_tests=self.config.use_statistical_tests,
            n_jobs=self.config.n_jobs,
            verbose=verbose,
            min_blocks_before_elimination=2,
        )

        # Prepare initial configs for this phase: extract active-role params
        phase_initial_configs: list[dict[str, Any]] = []
        for ic in self.initial_configs:
            phase_cfg = {k: v for k, v in ic.items() if k in active_params}
            for name, spec in context_params.items():
                phase_cfg.setdefault(name, spec.choices[0])
            if phase_cfg:
                phase_initial_configs.append(phase_cfg)

        tuner = RacingTuner(
            task=phase_task,
            scenario=scenario,
            seed=int(self.rng.integers(0, 2**31)),
            max_initial_configs=self.config.max_initial_configs,
            initial_configs=phase_initial_configs or None,
        )

        best_config_raw, history_raw = tuner.run(phase_eval_fn, verbose=verbose)
        best_config = {k: v for k, v in best_config_raw.items() if k in active_params}

        history: list[TrialResult] = []
        for trial in history_raw:
            trial_active = {k: v for k, v in trial.config.items() if k in active_params}
            merged_cfg = self._complete_config({**frozen_params, **trial_active}, full_param_space)
            details = dict(trial.details)
            details["phase"] = phase_name
            history.append(
                TrialResult(
                    trial_id=trial.trial_id,
                    config=merged_cfg,
                    score=float(trial.score),
                    details=details,
                )
            )

        diagnostics = self._build_phase_diagnostics(history)
        if diagnostics and verbose:
            d = diagnostics[-1]
            _logger().info(
                "[moea-tuner] Phase %s diagnostic: issue=%s suggestion=%s confidence=%.2f",
                phase_name,
                d.issue,
                d.suggestion,
                d.confidence,
            )

        return PhaseResult(
            phase=phase_name,
            best_config=best_config,
            history=history,
            diagnostics=diagnostics,
        )

    @property
    def phase_results(self) -> list[PhaseResult]:
        """Access results from each phase."""
        return list(self._phase_results)

    def _build_condition_context_params(
        self,
        *,
        conditions: list[Any],
        active_params: dict[str, ParamType],
        frozen_params: dict[str, Any],
        full_param_space: ParamSpace,
    ) -> dict[str, Categorical]:
        context: dict[str, Categorical] = {}
        for condition in conditions:
            for dep_name in _extract_cfg_keys(condition.expr):
                if dep_name in active_params or dep_name in context:
                    continue
                dep_spec = full_param_space.params.get(dep_name)
                if dep_spec is None:
                    continue
                dep_value = frozen_params.get(dep_name, _default_value(dep_spec))
                dep_role = getattr(dep_spec, "role", "structural")
                context[dep_name] = Categorical(dep_name, [dep_value], role=dep_role)
        return context

    def _complete_config(
        self,
        config: dict[str, Any],
        full_param_space: ParamSpace,
    ) -> dict[str, Any]:
        merged = dict(config)
        max_passes = len(full_param_space.params) + 1
        for _ in range(max_passes):
            added_any = False
            for name, spec in full_param_space.params.items():
                if name in merged:
                    continue
                if full_param_space.is_active(name, merged):
                    merged[name] = _default_value(spec)
                    added_any = True
            if not added_any:
                break
        return merged

    def _build_phase_diagnostics(self, history: list[TrialResult]) -> list[ConvergenceDiagnostic]:
        finite_scores = np.asarray([float(t.score) for t in history if np.isfinite(float(t.score))], dtype=float)
        if finite_scores.size < 2:
            return []
        if not self.task.maximize:
            finite_scores = -finite_scores
        progress_curve = np.maximum.accumulate(finite_scores)
        profile = compute_profile(progress_curve)
        return [diagnose(profile)]

    def _seed_initial_configs_from_diagnostics(
        self,
        *,
        phase_result: PhaseResult,
        frozen_params: dict[str, Any],
        role_groups: dict[str, list[ParamType]],
    ) -> None:
        if not phase_result.diagnostics:
            return
        diag = phase_result.diagnostics[-1]
        if diag.confidence < 0.5:
            return
        candidate_params = [p for p in role_groups.get(diag.suggestion, []) if p.name in frozen_params]
        if not candidate_params:
            return

        synthetic_cfgs: list[dict[str, Any]] = []
        n_new = min(3, len(candidate_params))
        for _ in range(n_new):
            param = candidate_params[int(self.rng.integers(0, len(candidate_params)))]
            cfg = dict(frozen_params)
            cfg[param.name] = _perturb_value(param, cfg[param.name], self.rng)
            synthetic_cfgs.append(cfg)
        self.initial_configs.extend(synthetic_cfgs)


def _brief_config(config: dict[str, Any], max_items: int = 6) -> str:
    """Format a config dict briefly for logging."""
    items = list(config.items())[:max_items]
    parts = [f"{k}={v}" for k, v in items]
    if len(config) > max_items:
        parts.append(f"... +{len(config) - max_items} more")
    return "{" + ", ".join(parts) + "}"


def _allocate_phase_experiments(
    *,
    total: int,
    split: tuple[float, float, float],
) -> list[int]:
    if total <= 0:
        raise ValueError("total must be > 0")
    n_phases = len(split)
    if total < n_phases:
        alloc = [0] * n_phases
        for i in range(total):
            alloc[i] = 1
        return alloc

    raw = [total * frac for frac in split]
    alloc = [int(math.floor(v)) for v in raw]
    remaining = total - sum(alloc)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - alloc[i], reverse=True)
    for i in order[:remaining]:
        alloc[i] += 1

    if total >= len(alloc):
        for i, value in enumerate(alloc):
            if value > 0:
                continue
            donor = max(range(len(alloc)), key=lambda idx: alloc[idx])
            if alloc[donor] <= 1:
                continue
            alloc[donor] -= 1
            alloc[i] = 1
    return alloc


def _extract_cfg_keys(expr: str) -> list[str]:
    return [m.group(1) for m in re.finditer(_CONDITION_CFG_KEY_PATTERN, expr)]


def _default_value(spec: ParamType) -> Any:
    if isinstance(spec, Real):
        if spec.log and spec.low > 0 and spec.high > 0:
            return float(math.exp(0.5 * (math.log(spec.low) + math.log(spec.high))))
        return float(0.5 * (spec.low + spec.high))
    if isinstance(spec, Int):
        if spec.log and spec.low > 0 and spec.high > 0:
            return int(round(math.exp(0.5 * (math.log(spec.low) + math.log(spec.high)))))
        return int(round(0.5 * (spec.low + spec.high)))
    if isinstance(spec, Categorical):
        return spec.choices[0]
    if isinstance(spec, Boolean):
        return False
    return None


def _perturb_value(spec: ParamType, value: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, Real):
        span = float(spec.high - spec.low)
        if span <= 0:
            return float(value)
        step = 0.15 * span
        perturbed = float(value) + float(rng.normal(0.0, step))
        return float(max(spec.low, min(spec.high, perturbed)))
    if isinstance(spec, Int):
        span = max(1, int(spec.high - spec.low))
        step = max(1, int(round(0.15 * span)))
        delta = int(rng.integers(-step, step + 1))
        perturbed = int(value) + delta
        return int(max(spec.low, min(spec.high, perturbed)))
    if isinstance(spec, Categorical):
        choices = list(spec.choices)
        if len(choices) <= 1:
            return value
        alternatives = [c for c in choices if c != value]
        return alternatives[int(rng.integers(0, len(alternatives)))]
    if isinstance(spec, Boolean):
        return not bool(value)
    return value


def _reindex_history(history: list[TrialResult], start_trial_id: int) -> list[TrialResult]:
    return [
        TrialResult(
            trial_id=start_trial_id + i,
            config=dict(trial.config),
            score=float(trial.score),
            details=dict(trial.details),
        )
        for i, trial in enumerate(history)
    ]


__all__ = ["MOEATuner", "MOEATunerConfig", "PhaseResult"]

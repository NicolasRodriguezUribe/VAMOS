"""
Controller for coordinating AOS policies and tracking rewards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import AdaptiveOperatorSelectionConfig
from .policies import OperatorBanditPolicy
from .portfolio import OperatorArm, OperatorPortfolio
from .rewards import RewardSummary, aggregate_reward, nd_insertion_rate, survival_rate


@dataclass(frozen=True)
class TraceRow:
    step: int
    mating_id: int
    op_id: str
    op_name: str
    reward: float
    reward_survival: float
    reward_nd_insertions: float
    reward_hv_delta: float
    batch_size: int


@dataclass(frozen=True)
class SummaryRow:
    op_id: str
    op_name: str
    pulls: int
    mean_reward: float
    total_reward: float
    usage_fraction: float


@dataclass
class AOSController:
    """
    AOS controller that tracks per-generation usage and updates the policy.
    """

    config: AdaptiveOperatorSelectionConfig
    portfolio: OperatorPortfolio
    policy: OperatorBanditPolicy
    _current_step: int | None = field(default=None, init=False)
    _trace_rows: list[TraceRow] = field(default_factory=list, init=False)
    _selections: list[tuple[int, int, int]] = field(default_factory=list, init=False)
    _gen_offspring: list[int] = field(default_factory=list, init=False)
    _gen_survivors: list[int] = field(default_factory=list, init=False)
    _gen_nd_insertions: list[int] = field(default_factory=list, init=False)
    _gen_pulls: list[int] = field(default_factory=list, init=False)
    _total_pulls: list[int] = field(default_factory=list, init=False)
    _total_reward: list[float] = field(default_factory=list, init=False)
    _eliminated: set[int] = field(default_factory=set, init=False)
    _generation_count: int = field(default=0, init=False)
    _reward_history_all: list[list[float]] = field(default_factory=list, init=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_arms = len(self.portfolio)
        if n_arms <= 0:
            raise ValueError("AOSController requires a non-empty portfolio.")
        if self.config.floor_prob < 0.0 or self.config.floor_prob > 1.0:
            raise ValueError("floor_prob must be within [0, 1].")
        self._gen_offspring = [0] * n_arms
        self._gen_survivors = [0] * n_arms
        self._gen_nd_insertions = [0] * n_arms
        self._gen_pulls = [0] * n_arms
        self._total_pulls = [0] * n_arms
        self._total_reward = [0.0] * n_arms
        self._eliminated = set()
        self._generation_count = 0
        self._reward_history_all = [[] for _ in range(n_arms)]
        self._rng = random.Random(self.config.rng_seed)

    def start_generation(self, step: int) -> None:
        self._current_step = int(step)
        for idx in range(len(self.portfolio)):
            self._gen_offspring[idx] = 0
            self._gen_survivors[idx] = 0
            self._gen_nd_insertions[idx] = 0
            self._gen_pulls[idx] = 0
        self._selections.clear()

    def select_arm(self, mating_id: int, batch_size: int) -> OperatorArm:
        if self._current_step is None:
            raise RuntimeError("start_generation() must be called before select_arm().")
        excluded = self._eliminated or None
        warm_idx = self._warmup_index()
        if warm_idx is not None:
            idx = warm_idx
        elif self.config.floor_prob > 0.0 and self._rng.random() < self.config.floor_prob:
            active = [i for i in range(len(self.portfolio)) if i not in self._eliminated]
            idx = self._rng.choice(active) if active else self._rng.randrange(len(self.portfolio))
        else:
            idx = self.policy.select_arm(excluded=excluded)
        self._selections.append((int(mating_id), idx, int(batch_size)))
        self._gen_pulls[idx] += 1
        self._total_pulls[idx] += 1
        return self.portfolio[idx]

    def observe_offspring(self, op_id: str, n: int) -> None:
        self._add_count(self._gen_offspring, op_id, n)

    def observe_survivors(self, op_id: str, n: int) -> None:
        self._add_count(self._gen_survivors, op_id, n)

    def observe_nd_insertions(self, op_id: str, n: int) -> None:
        self._add_count(self._gen_nd_insertions, op_id, n)

    def finalize_generation(self, step: int, evals: int | None = None, hv_delta_rate: float = 0.0) -> list[TraceRow]:
        if self._current_step is None:
            raise RuntimeError("start_generation() must be called before finalize_generation().")
        if int(step) != self._current_step:
            raise ValueError("finalize_generation() step does not match current generation.")

        reward_by_arm: dict[int, RewardSummary] = {}
        for idx, arm in enumerate(self.portfolio):
            if self._gen_pulls[idx] <= 0:
                continue
            n_off = self._gen_offspring[idx]
            n_surv = self._gen_survivors[idx]
            n_nd = self._gen_nd_insertions[idx]
            surv_rate = survival_rate(n_surv, n_off)
            nd_rate = nd_insertion_rate(n_nd, n_off)
            summary = aggregate_reward(
                self.config.reward_scope,
                surv_rate,
                nd_rate,
                hv_delta_rate=float(hv_delta_rate),
                weights=self.config.reward_weights,
            )
            reward_by_arm[idx] = summary
            self.policy.update(idx, summary.reward)
            self._total_reward[idx] += summary.reward * self._gen_pulls[idx]
            self._reward_history_all[idx].append(summary.reward)

        rows: list[TraceRow] = []
        for mating_id, idx, batch_size in self._selections:
            arm = self.portfolio[idx]
            row_summary = reward_by_arm.get(idx)
            if row_summary is None:
                row_summary = RewardSummary(0.0, 0.0, 0.0, 0.0)
            rows.append(
                TraceRow(
                    step=self._current_step,
                    mating_id=mating_id,
                    op_id=arm.op_id,
                    op_name=arm.name,
                    reward=row_summary.reward,
                    reward_survival=row_summary.reward_survival,
                    reward_nd_insertions=row_summary.reward_nd_insertions,
                    reward_hv_delta=row_summary.reward_hv_delta,
                    batch_size=batch_size,
                )
            )

        self._trace_rows.extend(rows)
        self._generation_count += 1

        # --- Arm elimination ---
        elim_after = self.config.elimination_after
        if elim_after > 0 and self._generation_count == elim_after:
            self._try_eliminate_arms()

        return rows

    def _try_eliminate_arms(self) -> None:
        """Eliminate arms whose mean reward is significantly below the best arm."""
        n_arms = len(self.portfolio)
        min_arms = max(1, self.config.elimination_min_arms)
        active = [i for i in range(n_arms) if i not in self._eliminated]
        if len(active) <= min_arms:
            return

        # Compute mean and std of rewards for each active arm
        stats: dict[int, tuple[float, float, int]] = {}  # idx -> (mean, std, count)
        for idx in active:
            rewards = self._reward_history_all[idx]
            if not rewards:
                stats[idx] = (0.0, 0.0, 0)
                continue
            n = len(rewards)
            mean = sum(rewards) / n
            if n > 1:
                var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            stats[idx] = (mean, std, n)

        # Find the best arm by mean reward
        best_idx = max(active, key=lambda i: stats[i][0])
        best_mean, best_std, best_n = stats[best_idx]

        # Eliminate arms that are z-score below the best
        z_threshold = self.config.elimination_z
        to_eliminate: list[int] = []
        for idx in active:
            if idx == best_idx:
                continue
            mean_i, std_i, n_i = stats[idx]
            if n_i < 2 or best_n < 2:
                continue
            # Pooled standard error of the difference in means
            se = math.sqrt((best_std ** 2 / best_n) + (std_i ** 2 / n_i))
            if se <= 0.0:
                continue
            z = (best_mean - mean_i) / se
            if z >= z_threshold:
                to_eliminate.append(idx)

        # Never eliminate below min_arms
        remaining = len(active) - len(to_eliminate)
        if remaining < min_arms:
            # Sort by z-score (worst first) and only eliminate enough to keep min_arms
            to_eliminate.sort(key=lambda i: stats[i][0])
            to_eliminate = to_eliminate[: len(active) - min_arms]

        self._eliminated.update(to_eliminate)

    def summary_rows(self) -> list[SummaryRow]:
        total_pulls = sum(self._total_pulls)
        rows: list[SummaryRow] = []
        for idx, arm in enumerate(self.portfolio):
            pulls = self._total_pulls[idx]
            total_reward = self._total_reward[idx]
            mean_reward = total_reward / pulls if pulls > 0 else 0.0
            usage_fraction = pulls / total_pulls if total_pulls > 0 else 0.0
            rows.append(
                SummaryRow(
                    op_id=arm.op_id,
                    op_name=arm.name,
                    pulls=pulls,
                    mean_reward=mean_reward,
                    total_reward=total_reward,
                    usage_fraction=usage_fraction,
                )
            )
        return rows

    def _add_count(self, target: list[int], op_id: str, n: int) -> None:
        if n < 0:
            raise ValueError("Count increments must be non-negative.")
        if n == 0:
            return
        idx = self.portfolio.index_of(op_id)
        target[idx] += int(n)

    def _warmup_index(self) -> int | None:
        min_usage = self.config.min_usage
        if min_usage <= 0:
            return None
        counts = self.policy.counts()
        active = [(idx, counts[idx]) for idx in range(len(counts)) if idx not in self._eliminated]
        if not active:
            return None
        min_count = min(c for _, c in active)
        if min_count >= min_usage:
            return None
        for idx, count in active:
            if count == min_count:
                return idx
        return None


__all__ = ["AOSController", "TraceRow", "SummaryRow"]

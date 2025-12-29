"""
Controller for coordinating AOS policies and tracking rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    def __post_init__(self) -> None:
        n_arms = len(self.portfolio)
        if n_arms <= 0:
            raise ValueError("AOSController requires a non-empty portfolio.")
        self._gen_offspring = [0] * n_arms
        self._gen_survivors = [0] * n_arms
        self._gen_nd_insertions = [0] * n_arms
        self._gen_pulls = [0] * n_arms
        self._total_pulls = [0] * n_arms
        self._total_reward = [0.0] * n_arms

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
        idx = self.policy.select_arm()
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

    def finalize_generation(self, step: int, evals: int | None = None) -> list[TraceRow]:
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
                hv_delta_rate=0.0,
                weights=self.config.reward_weights,
            )
            reward_by_arm[idx] = summary
            self.policy.update(idx, summary.reward)
            self._total_reward[idx] += summary.reward * self._gen_pulls[idx]

        rows: list[TraceRow] = []
        for mating_id, idx, batch_size in self._selections:
            arm = self.portfolio[idx]
            summary = reward_by_arm.get(idx)
            if summary is None:
                summary = RewardSummary(0.0, 0.0, 0.0, 0.0)
            rows.append(
                TraceRow(
                    step=self._current_step,
                    mating_id=mating_id,
                    op_id=arm.op_id,
                    op_name=arm.name,
                    reward=summary.reward,
                    reward_survival=summary.reward_survival,
                    reward_nd_insertions=summary.reward_nd_insertions,
                    reward_hv_delta=summary.reward_hv_delta,
                    batch_size=batch_size,
                )
            )

        self._trace_rows.extend(rows)
        return rows

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


__all__ = ["AOSController", "TraceRow", "SummaryRow"]

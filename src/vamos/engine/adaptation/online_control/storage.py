from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .contracts import Credit, HierarchicalAction, Outcome, SearchState


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _switch_count(values: list[str]) -> int:
    if len(values) < 2:
        return 0
    switches = 0
    previous = values[0]
    for current in values[1:]:
        if current != previous:
            switches += 1
        previous = current
    return switches


def _share_map(values: list[str]) -> dict[str, float]:
    if not values:
        return {}
    counts = Counter(values)
    total = float(len(values))
    return {key: float(count / total) for key, count in sorted(counts.items())}


@dataclass(frozen=True)
class TraceRow:
    step_index: int
    search_state: SearchState
    action: HierarchicalAction
    outcome: Outcome
    credit: Credit

    def as_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "generation": self.search_state.generation,
            "regime": self.action.regime.value,
            "operator_family": self.action.operator_family.value,
            "intent_prototype": self.action.parametric_intent.prototype,
            "parametric_intent": self.action.parametric_intent.as_dict(),
            "outcome_summary": {
                "success": self.outcome.success,
                "reward_hint": self.outcome.reward_hint,
                "accepted_ratio": self.outcome.accepted_ratio if self.outcome.accepted_ratio is not None else self.outcome.survivor_ratio,
                "nd_insertions_ratio": self.outcome.nd_insertions_ratio,
                "feasible_delta": self.outcome.feasible_delta,
                "extent_gain": self.outcome.extent_gain,
                "diversity_gain": self.outcome.diversity_gain if self.outcome.diversity_gain is not None else self.outcome.diversity_delta,
                "quality_delta": self.outcome.quality_delta,
                "overhead_ms": self.outcome.overhead_ms,
            },
            "reward": self.credit.reward,
            "bounded_reward": self.credit.bounded_reward,
            "search_state": self.search_state.as_dict(),
            "action": self.action.as_dict(),
            "outcome": self.outcome.as_dict(),
            "credit": self.credit.as_dict(),
        }

    def to_flat_dict(self) -> dict[str, object]:
        accepted_ratio = self.outcome.accepted_ratio if self.outcome.accepted_ratio is not None else self.outcome.survivor_ratio
        diversity_gain = self.outcome.diversity_gain if self.outcome.diversity_gain is not None else self.outcome.diversity_delta
        return {
            "host": self.search_state.host,
            "step_index": self.step_index,
            "generation": self.search_state.generation,
            "evaluations": self.search_state.evaluations,
            "budget_progress": self.search_state.budget_progress,
            "regime": self.action.regime.value,
            "operator_family": self.action.operator_family.value,
            "intent_prototype": self.action.parametric_intent.prototype,
            "exploration_strength": self.action.parametric_intent.exploration_strength,
            "locality": self.action.parametric_intent.locality,
            "mutation_strength": self.action.parametric_intent.mutation_strength,
            "feasibility_bias": self.action.parametric_intent.feasibility_bias,
            "policy": self.action.metadata.get("policy"),
            "selection_mode": self.action.metadata.get("selection_mode"),
            "success": self.outcome.success,
            "accepted_ratio": accepted_ratio,
            "nd_insertions_ratio": self.outcome.nd_insertions_ratio,
            "feasible_ratio": self.outcome.feasible_ratio,
            "feasible_delta": self.outcome.feasible_delta,
            "extent": self.outcome.extent,
            "extent_gain": self.outcome.extent_gain,
            "diversity": self.outcome.diversity,
            "diversity_gain": diversity_gain,
            "quality_indicator": self.outcome.quality_indicator,
            "quality_delta": self.outcome.quality_delta,
            "overhead_ms": self.outcome.overhead_ms,
            "reward_hint": self.outcome.reward_hint,
            "reward": self.credit.reward,
            "bounded_reward": self.credit.bounded_reward,
            "credit_model": self.credit.metadata.get("model"),
        }


@dataclass
class InMemoryTraceStore:
    enabled: bool = True
    _rows: list[TraceRow] = field(default_factory=list)

    def append(self, row: TraceRow) -> None:
        if self.enabled:
            self._rows.append(row)

    def rows(self) -> list[TraceRow]:
        return list(self._rows)

    def to_dicts(self) -> list[dict[str, object]]:
        return [row.as_dict() for row in self._rows]

    def to_flat_dicts(self) -> list[dict[str, object]]:
        return [row.to_flat_dict() for row in self._rows]

    def summary_rows(self) -> list[dict[str, object]]:
        rows = self._rows
        if not rows:
            return []

        grouped: dict[tuple[str, str], list[TraceRow]] = defaultdict(list)
        for row in rows:
            grouped[("regime", row.action.regime.value)].append(row)
            grouped[("operator_family", row.action.operator_family.value)].append(row)
            grouped[("intent_prototype", row.action.parametric_intent.prototype or "unspecified")].append(row)

        total = float(len(rows))
        summary_rows: list[dict[str, object]] = []
        for (summary_type, label), bucket in sorted(grouped.items()):
            share = len(bucket) / total
            bounded_rewards = [trace.credit.bounded_reward for trace in bucket]
            rewards = [trace.credit.reward for trace in bucket]
            overhead = [float(trace.outcome.overhead_ms) for trace in bucket if trace.outcome.overhead_ms is not None]
            payload = {
                "summary_type": summary_type,
                "label": label,
                "count": len(bucket),
                "share": share,
                "average_reward": _mean(rewards),
                "average_bounded_reward": _mean(bounded_rewards),
                "average_overhead_ms": _mean(overhead),
            }
            if summary_type == "regime":
                payload["selected_regime_share"] = share
            elif summary_type == "operator_family":
                payload["selected_family_share"] = share
            else:
                payload["selected_intent_share"] = share
            summary_rows.append(payload)
        return summary_rows

    def run_summary(self) -> dict[str, object]:
        rows = self._rows
        if not rows:
            return {
                "steps": 0,
                "average_reward": 0.0,
                "average_bounded_reward": 0.0,
                "average_overhead_ms": 0.0,
                "regime_switches": 0,
                "family_switches": 0,
                "intent_switches": 0,
                "regime_shares": {},
                "family_shares": {},
                "intent_shares": {},
                "regime_concentration": 0.0,
                "family_concentration": 0.0,
                "intent_concentration": 0.0,
            }

        regimes = [row.action.regime.value for row in rows]
        families = [row.action.operator_family.value for row in rows]
        intents = [row.action.parametric_intent.prototype or "unspecified" for row in rows]
        rewards = [row.credit.reward for row in rows]
        bounded_rewards = [row.credit.bounded_reward for row in rows]
        overhead = [float(row.outcome.overhead_ms) for row in rows if row.outcome.overhead_ms is not None]
        family_shares = _share_map(families)
        regime_shares = _share_map(regimes)
        intent_shares = _share_map(intents)

        return {
            "host": rows[0].search_state.host,
            "policy": rows[0].action.metadata.get("policy"),
            "credit_model": rows[0].credit.metadata.get("model"),
            "steps": len(rows),
            "average_reward": _mean(rewards),
            "average_bounded_reward": _mean(bounded_rewards),
            "average_overhead_ms": _mean(overhead),
            "regime_switches": _switch_count(regimes),
            "family_switches": _switch_count(families),
            "intent_switches": _switch_count(intents),
            "regime_shares": regime_shares,
            "family_shares": family_shares,
            "intent_shares": intent_shares,
            "dominant_regime": max(regime_shares, key=regime_shares.get),
            "dominant_family": max(family_shares, key=family_shares.get),
            "dominant_intent": max(intent_shares, key=intent_shares.get),
            "regime_concentration": sum(share * share for share in regime_shares.values()),
            "family_concentration": sum(share * share for share in family_shares.values()),
            "intent_concentration": sum(share * share for share in intent_shares.values()),
        }


__all__ = ["InMemoryTraceStore", "TraceRow"]

"""
Adaptive Operator Selection (AOS) primitives.
"""

from .config import AdaptiveOperatorSelectionConfig
from .controller import AOSController, SummaryRow, TraceRow
from .logging import write_aos_summary, write_aos_trace
from .policies import EpsGreedyPolicy, EXP3Policy, OperatorBanditPolicy, SlidingWindowUCBPolicy, ThompsonSamplingPolicy, UCBPolicy
from .portfolio import OperatorArm, OperatorPortfolio
from .rewards import RewardSummary, aggregate_reward, nd_insertion_rate, survival_rate

__all__ = [
    "AdaptiveOperatorSelectionConfig",
    "AOSController",
    "SummaryRow",
    "TraceRow",
    "write_aos_summary",
    "write_aos_trace",
    "OperatorBanditPolicy",
    "UCBPolicy",
    "EpsGreedyPolicy",
    "EXP3Policy",
    "ThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
    "OperatorArm",
    "OperatorPortfolio",
    "RewardSummary",
    "aggregate_reward",
    "nd_insertion_rate",
    "survival_rate",
]

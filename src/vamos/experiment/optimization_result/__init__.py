"""Public optimization result types and helpers."""

from .model import BestResult, OptimizationResult, TopKResult
from .ranking import BestMethod, RankingMethod, RankingSource

__all__ = ["BestMethod", "BestResult", "OptimizationResult", "RankingMethod", "RankingSource", "TopKResult"]

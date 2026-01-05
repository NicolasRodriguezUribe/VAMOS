"""
Configuration helpers for adaptive operator selection (AOS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULT_REWARD_WEIGHTS = {
    "survival": 0.5,
    "nd_insertions": 0.5,
    "hv_delta": 0.0,
}


def _normalize_reward_weights(raw: Mapping[str, Any] | None) -> dict[str, float]:
    weights = dict(DEFAULT_REWARD_WEIGHTS)
    if not raw:
        return weights
    if raw.get("survival") is not None:
        weights["survival"] = float(raw["survival"])
    if raw.get("nd_insertions") is not None:
        weights["nd_insertions"] = float(raw["nd_insertions"])
    if raw.get("hv_delta") is not None:
        weights["hv_delta"] = float(raw["hv_delta"])
    return weights


@dataclass(frozen=True)
class AdaptiveOperatorSelectionConfig:
    """
    Configuration for adaptive operator selection.

    The config is intentionally minimal and decoupled from algorithms. It is
    designed to be passed into an AOSController with a portfolio and policy.
    """

    enabled: bool = False
    method: str = "epsilon_greedy"
    epsilon: float = 0.1
    c: float = 1.0
    gamma: float = 0.2
    min_usage: int = 1
    rng_seed: int | None = None
    window_size: int = 0  # For sliding window policies; 0 = disabled
    reward_scope: str = "combined"
    reward_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | None) -> "AdaptiveOperatorSelectionConfig":
        """
        Create a config instance from a dictionary.
        """
        if not config:
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            method=str(config.get("method", "epsilon_greedy")),
            epsilon=float(config.get("epsilon", 0.1)),
            c=float(config.get("c", 1.0)),
            gamma=float(config.get("gamma", 0.2)),
            min_usage=int(config.get("min_usage", 1)),
            rng_seed=config.get("rng_seed"),
            window_size=int(config.get("window_size", 0)),
            reward_scope=str(config.get("reward_scope", "combined")),
            reward_weights=_normalize_reward_weights(config.get("reward_weights")),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config to a JSON-serializable dictionary.
        """
        return {
            "enabled": self.enabled,
            "method": self.method,
            "epsilon": self.epsilon,
            "c": self.c,
            "gamma": self.gamma,
            "min_usage": self.min_usage,
            "rng_seed": self.rng_seed,
            "window_size": self.window_size,
            "reward_scope": self.reward_scope,
            "reward_weights": dict(self.reward_weights),
        }


__all__ = ["AdaptiveOperatorSelectionConfig", "DEFAULT_REWARD_WEIGHTS"]

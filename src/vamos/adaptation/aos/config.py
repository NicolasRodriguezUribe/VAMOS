"""
Configuration helpers for adaptive operator selection (AOS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
from collections.abc import Mapping

DEFAULT_REWARD_WEIGHTS = {
    "survival": 0.5,
    "nd_insertions": 0.5,
    "hv_delta": 0.0,
}

AOS_CONFIG_KEYS = {
    "enabled",
    "method",
    "epsilon",
    "c",
    "gamma",
    "min_usage",
    "rng_seed",
    "window_size",
    "reward_scope",
    "reward_weights",
    "hv_reference_point",
    "hv_reference_hv",
    "floor_prob",
    "operator_pool",
    "elimination_after",
    "elimination_z",
    "elimination_min_arms",
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
    hv_reference_point: tuple[float, ...] | None = None
    hv_reference_hv: float | None = None
    floor_prob: float = 0.0
    elimination_after: int = 0  # Generations before arm elimination starts; 0 = disabled
    elimination_z: float = 1.5  # Z-score threshold: eliminate arms this many stdevs below best
    elimination_min_arms: int = 2  # Never eliminate below this many active arms

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | None) -> AdaptiveOperatorSelectionConfig:
        """
        Create a config instance from a dictionary.
        """
        if not config:
            return cls()
        unexpected = set(config) - AOS_CONFIG_KEYS
        if unexpected:
            raise ValueError(f"Unsupported AOS config keys: {sorted(unexpected)}")
        floor_prob = float(config.get("floor_prob", 0.0))
        if floor_prob < 0.0 or floor_prob > 1.0:
            raise ValueError("floor_prob must be within [0, 1].")
        window_size = int(config.get("window_size", 0))
        if window_size < 0:
            raise ValueError("window_size must be >= 0.")
        hv_ref_point = config.get("hv_reference_point")
        if hv_ref_point is None:
            hv_reference_point = None
        else:
            if not isinstance(hv_ref_point, (list, tuple)):
                raise TypeError("hv_reference_point must be a list or tuple of floats.")
            hv_reference_point = tuple(float(x) for x in hv_ref_point)
            if not hv_reference_point or any(not math.isfinite(x) for x in hv_reference_point):
                raise ValueError("hv_reference_point must contain finite floats.")
        hv_ref_hv = config.get("hv_reference_hv")
        hv_reference_hv = None if hv_ref_hv is None else float(hv_ref_hv)
        if hv_reference_hv is not None and hv_reference_hv <= 0.0:
            raise ValueError("hv_reference_hv must be > 0 when provided.")
        elimination_after = int(config.get("elimination_after", 0))
        if elimination_after < 0:
            raise ValueError("elimination_after must be >= 0.")
        elimination_z = float(config.get("elimination_z", 1.5))
        if elimination_z < 0.0:
            raise ValueError("elimination_z must be >= 0.")
        elimination_min_arms = int(config.get("elimination_min_arms", 2))
        if elimination_min_arms < 1:
            raise ValueError("elimination_min_arms must be >= 1.")
        return cls(
            enabled=bool(config.get("enabled", False)),
            method=str(config.get("method", "epsilon_greedy")).lower(),
            epsilon=float(config.get("epsilon", 0.1)),
            c=float(config.get("c", 1.0)),
            gamma=float(config.get("gamma", 0.2)),
            min_usage=int(config.get("min_usage", 1)),
            rng_seed=config.get("rng_seed"),
            window_size=window_size,
            reward_scope=str(config.get("reward_scope", "combined")).lower(),
            reward_weights=_normalize_reward_weights(config.get("reward_weights")),
            hv_reference_point=hv_reference_point,
            hv_reference_hv=hv_reference_hv,
            floor_prob=floor_prob,
            elimination_after=elimination_after,
            elimination_z=elimination_z,
            elimination_min_arms=elimination_min_arms,
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
            "hv_reference_point": list(self.hv_reference_point) if self.hv_reference_point is not None else None,
            "hv_reference_hv": self.hv_reference_hv,
            "floor_prob": self.floor_prob,
            "elimination_after": self.elimination_after,
            "elimination_z": self.elimination_z,
            "elimination_min_arms": self.elimination_min_arms,
        }


__all__ = ["AdaptiveOperatorSelectionConfig", "DEFAULT_REWARD_WEIGHTS"]

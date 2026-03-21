from __future__ import annotations

from .contracts import ParametricIntent

DEFAULT_PROTOTYPE_SET = "default"

_PROTOTYPE_VALUES: dict[str, dict[str, float]] = {
    "exploratory": {
        "exploration_strength": 0.88,
        "locality": 0.20,
        "mutation_strength": 0.62,
        "feasibility_bias": 0.28,
    },
    "balanced": {
        "exploration_strength": 0.50,
        "locality": 0.50,
        "mutation_strength": 0.50,
        "feasibility_bias": 0.50,
    },
    "local_refine": {
        "exploration_strength": 0.18,
        "locality": 0.88,
        "mutation_strength": 0.24,
        "feasibility_bias": 0.58,
    },
    "mutation_heavy": {
        "exploration_strength": 0.58,
        "locality": 0.34,
        "mutation_strength": 0.92,
        "feasibility_bias": 0.36,
    },
    "feasibility_biased": {
        "exploration_strength": 0.34,
        "locality": 0.72,
        "mutation_strength": 0.56,
        "feasibility_bias": 0.96,
    },
}


def normalize_prototype_set(name: str | None) -> str:
    normalized = str(name or DEFAULT_PROTOTYPE_SET).strip().lower()
    if normalized != DEFAULT_PROTOTYPE_SET:
        raise ValueError("online_control.prototype_set must be 'default'.")
    return normalized


def available_intent_prototypes(prototype_set: str | None = None) -> tuple[str, ...]:
    normalize_prototype_set(prototype_set)
    return tuple(_PROTOTYPE_VALUES.keys())


def build_intent_prototype(name: str, *, prototype_set: str | None = None) -> ParametricIntent:
    normalize_prototype_set(prototype_set)
    key = str(name).strip().lower()
    if key not in _PROTOTYPE_VALUES:
        allowed = ", ".join(sorted(_PROTOTYPE_VALUES))
        raise ValueError(f"Unknown online_control intent prototype '{name}'. Expected one of: {allowed}.")
    payload = dict(_PROTOTYPE_VALUES[key])
    return ParametricIntent(
        prototype=key,
        exploration_strength=float(payload["exploration_strength"]),
        locality=float(payload["locality"]),
        mutation_strength=float(payload["mutation_strength"]),
        feasibility_bias=float(payload["feasibility_bias"]),
        metadata={"prototype": key, "prototype_set": DEFAULT_PROTOTYPE_SET},
    )


__all__ = [
    "DEFAULT_PROTOTYPE_SET",
    "available_intent_prototypes",
    "build_intent_prototype",
    "normalize_prototype_set",
]

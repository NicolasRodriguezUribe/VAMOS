from __future__ import annotations

from typing import Any

from vamos.engine.config.loader import load_experiment_spec

from .types import SpecDefaults


def load_spec_defaults(config_path: str | None) -> SpecDefaults:
    """Load YAML/JSON spec if provided and return defaults per algorithm and problems."""
    spec: dict[str, Any] = {}
    problem_overrides: dict[str, Any] = {}
    experiment_defaults: dict[str, Any] = {}
    nsgaii_defaults: dict[str, Any] = {}
    moead_defaults: dict[str, Any] = {}
    smsemoa_defaults: dict[str, Any] = {}
    nsgaiii_defaults: dict[str, Any] = {}
    if config_path:
        spec = load_experiment_spec(config_path)
        problem_overrides = spec.get("problems", {}) or {}
        experiment_defaults = spec.get("defaults", {}) or {k: v for k, v in spec.items() if k != "problems"}
        nsgaii_defaults = experiment_defaults.get("nsgaii", {}) or {}
        moead_defaults = experiment_defaults.get("moead", {}) or {}
        smsemoa_defaults = experiment_defaults.get("smsemoa", {}) or {}
        nsgaiii_defaults = experiment_defaults.get("nsgaiii", {}) or {}
    return SpecDefaults(
        spec=spec,
        problem_overrides=problem_overrides,
        experiment_defaults=experiment_defaults,
        nsgaii_defaults=nsgaii_defaults,
        moead_defaults=moead_defaults,
        smsemoa_defaults=smsemoa_defaults,
        nsgaiii_defaults=nsgaiii_defaults,
    )

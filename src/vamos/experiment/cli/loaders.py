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
        raw = load_experiment_spec(config_path)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("Experiment spec must be a mapping (YAML/JSON object).")
        spec = raw

        problems = spec.get("problems", {})
        if problems is None:
            problems = {}
        if not isinstance(problems, dict):
            raise ValueError("Experiment spec 'problems' must be a mapping of problem_key -> overrides.")
        problem_overrides = problems

        defaults = spec.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("Experiment spec 'defaults' must be a mapping when provided.")
        experiment_defaults = defaults

        nsgaii_defaults = experiment_defaults.get("nsgaii", {}) or {}
        if not isinstance(nsgaii_defaults, dict):
            raise ValueError("Experiment spec 'defaults.nsgaii' must be a mapping when provided.")
        moead_defaults = experiment_defaults.get("moead", {}) or {}
        if not isinstance(moead_defaults, dict):
            raise ValueError("Experiment spec 'defaults.moead' must be a mapping when provided.")
        smsemoa_defaults = experiment_defaults.get("smsemoa", {}) or {}
        if not isinstance(smsemoa_defaults, dict):
            raise ValueError("Experiment spec 'defaults.smsemoa' must be a mapping when provided.")
        nsgaiii_defaults = experiment_defaults.get("nsgaiii", {}) or {}
        if not isinstance(nsgaiii_defaults, dict):
            raise ValueError("Experiment spec 'defaults.nsgaiii' must be a mapping when provided.")
    return SpecDefaults(
        spec=spec,
        problem_overrides=problem_overrides,
        experiment_defaults=experiment_defaults,
        nsgaii_defaults=nsgaii_defaults,
        moead_defaults=moead_defaults,
        smsemoa_defaults=smsemoa_defaults,
        nsgaiii_defaults=nsgaiii_defaults,
    )

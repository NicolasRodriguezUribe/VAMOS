"""Base utilities for algorithm configuration."""

from __future__ import annotations

import dataclasses
import difflib
import json
import sys
import warnings
from dataclasses import asdict
from typing import Any, Literal, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# ── Shared Literal type aliases ──────────────────────────────────────
ResultMode = Literal["non_dominated", "population"]
ConstraintModeStr = Literal["none", "feasibility", "penalty", "epsilon"]
LiveCallbackMode = Literal["nd_only", "all"]
IndicatorType = Literal["eps", "hypervolume"]

# ── Known operator names (runtime registry + aliases) ────────────────
# Plain set literals (no frozenset() calls) to avoid import-time side effects.
_KNOWN_OPERATORS: dict[str, set[str]] = {
    "crossover": {
        "sbx",
        "blx_alpha",
        "blx_alpha_beta",
        "arithmetic",
        "whole_arithmetic",
        "laplace",
        "fuzzy",
        "de",
        "pcx",
        "undx",
        "simplex",
        "one_point",
        "two_point",
        "binary_uniform",
        "hux",
        "spx",
        "pmx",
        "cx",
        "cycle",
        "position_based",
        "erx",
        "aex",
        "ox",
        "order",
        "int_uniform",
        "int_arithmetic",
        "int_sbx",
        "mixed",
        # aliases from helpers
        "oxd",
        "position",
        "pos",
        "edge",
        "edge_recombination",
        "single_point",
        "1point",
        "blend",
        "uniform",
    },
    "mutation": {
        "polynomial",
        "non_uniform",
        "gaussian",
        "uniform_reset",
        "cauchy",
        "uniform",
        "linked_polynomial",
        "levy_flight",
        "power_law",
        "bitflip",
        "segment_inversion",
        "swap",
        "insert",
        "scramble",
        "inversion",
        "displacement",
        "two_opt",
        "reset",
        "int_pm",
        "creep",
        "boundary",
        "int_gaussian",
        "mixed_mutation",
        "mixed",
        # aliases from helpers
        "random_reset",
        "bit_flip",
    },
    "selection": {
        "tournament",
        "random",
        "boltzmann",
        "ranking",
        "sus",
    },
    "repair": {
        "clip",
        "clamp",
        "reflect",
        "random",
        "resample",
        "round",
    },
    "initializer": {
        "random",
        "lhs",
        "scatter",
        "scatter_search",
        "sobol",
        "halton",
        "obl",
        "opposition",
        "custom",
    },
}


def _validate_operator_name(name: str, category: str) -> None:
    """Validate an operator name against known operators.

    - Raises ``ValueError`` when a close match exists (likely a typo).
    - Issues a ``UserWarning`` for completely unknown names (possibly custom).
    """
    known = _KNOWN_OPERATORS.get(category)
    if known is None:
        return
    lower = name.strip().lower()
    if lower in known:
        return
    matches = difflib.get_close_matches(lower, sorted(known), n=3, cutoff=0.5)
    if matches:
        available = ", ".join(sorted(known))
        raise ValueError(f"Unknown {category} '{name}'. Did you mean: {', '.join(matches)}?\n  Available: {available}")
    warnings.warn(
        f"Unknown {category} '{name}'. If this is a custom operator, you can ignore this warning.",
        UserWarning,
        stacklevel=4,
    )


def _validate_operators(cfg: dict[str, Any]) -> None:
    """Validate all operator names found in a builder cfg dict."""
    for key in ("crossover", "mutation", "selection", "repair"):
        val = cfg.get(key)
        if val is not None and isinstance(val, tuple) and len(val) >= 1:
            _validate_operator_name(str(val[0]), key)
    init = cfg.get("initializer")
    if isinstance(init, dict) and "type" in init:
        _validate_operator_name(str(init["type"]), "initializer")


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    __dataclass_fields__: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(cast(Any, self)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Reconstruct a config from a serialised dict (inverse of ``to_dict()``)."""
        field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        filtered: dict[str, object] = {}
        for key, value in data.items():
            if key not in field_names:
                continue
            # Reconstruct nested ExternalArchiveConfig from plain dict
            if key == "external_archive" and isinstance(value, dict):
                from vamos.engine.archive import ExternalArchiveConfig

                value = ExternalArchiveConfig(**value)
            filtered[key] = value
        return cls(**filtered)

    @staticmethod
    def _format_field_value(val: object) -> str:
        """Format a config field value for display."""
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], dict):
            name, params = val
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                return f"{name}({param_str})"
            return str(name)
        return repr(val)

    def __repr__(self) -> str:
        """Show only fields that differ from defaults."""
        cls = type(self)
        fields = dataclasses.fields(cls)  # type: ignore[arg-type]
        parts: list[str] = []
        for f in fields:
            val = getattr(self, f.name)
            if f.default is not dataclasses.MISSING and val == f.default:
                continue
            if f.default_factory is not dataclasses.MISSING and val == f.default_factory():
                continue
            if val is None:
                continue
            parts.append(f"{f.name}={self._format_field_value(val)}")
        return f"{cls.__name__}({', '.join(parts)})"

    def describe(self) -> str:
        """Return a human-friendly summary of this configuration."""
        cls = type(self)
        fields = dataclasses.fields(cls)  # type: ignore[arg-type]
        algo_name = cls.__name__.replace("Config", "").upper()
        if algo_name == "NSGAII":
            algo_name = "NSGA-II"
        elif algo_name == "NSGAIII":
            algo_name = "NSGA-III"
        elif algo_name == "SMSEMOA":
            algo_name = "SMS-EMOA"
        elif algo_name == "AGEMOEA":
            algo_name = "AGE-MOEA"

        lines: list[str] = [f"{algo_name} Configuration"]
        lines.append("-" * len(lines[0]))

        for f in fields:
            val = getattr(self, f.name)
            if val is None:
                continue
            # Skip internal/callback fields from display
            if f.name in ("generation_callback_copy", "parent_selection_filter"):
                continue
            label = f.name.replace("_", " ").title()
            # Format operator tuples
            if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], dict):
                name, params = val
                if params:
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    formatted = f"{name} ({param_str})"
                else:
                    formatted = name
                lines.append(f"  {label:.<30s} {formatted}")
            elif isinstance(val, bool):
                lines.append(f"  {label:.<30s} {'yes' if val else 'no'}")
            elif callable(val):
                fn_name = getattr(val, "__name__", repr(val))
                lines.append(f"  {label:.<30s} {fn_name}")
            else:
                lines.append(f"  {label:.<30s} {val}")
        return "\n".join(lines)

    def diff(self, other: _SerializableConfig) -> str:
        """Compare this config with *other* and return a summary of differences."""
        cls = type(self)
        if type(other) is not cls:
            raise TypeError(f"Cannot diff {cls.__name__} with {type(other).__name__}")
        fields = dataclasses.fields(cls)  # type: ignore[arg-type]
        diffs: list[str] = []
        for f in fields:
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a != b:
                diffs.append(f"  {f.name}: {self._format_field_value(a)} -> {self._format_field_value(b)}")
        if not diffs:
            return "(no differences)"
        header = f"{cls.__name__} diff:"
        return "\n".join([header, *diffs])

    @staticmethod
    def available_operators(category: str | None = None) -> dict[str, list[str]]:
        """Return known operator names, optionally filtered by *category*.

        Args:
            category: One of ``"crossover"``, ``"mutation"``, ``"selection"``,
                ``"repair"``, ``"initializer"``.  ``None`` returns all categories.
        """
        if category is not None:
            key = category.strip().lower()
            known = _KNOWN_OPERATORS.get(key)
            if known is None:
                available = ", ".join(sorted(_KNOWN_OPERATORS))
                raise ValueError(f"Unknown category '{category}'. Available: {available}")
            return {key: sorted(known)}
        return {k: sorted(v) for k, v in _KNOWN_OPERATORS.items()}

    @staticmethod
    def default_operators(encoding: str = "real") -> dict[str, tuple[str, dict[str, Any]]]:
        """Return the default crossover/mutation operators for *encoding*.

        >>> NSGAIIConfig.default_operators("permutation")
        {'crossover': ('ox', {}), 'mutation': ('swap', {'prob': 0.1})}
        """
        cx, mt = _default_operators_for_encoding(encoding, mut_prob=0.1)
        return {"crossover": cx, "mutation": mt}


def _default_operators_for_encoding(
    encoding: str,
    mut_prob: float,
) -> tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]]]:
    """Return (crossover_tuple, mutation_tuple) defaults for the given encoding."""
    from vamos.foundation.encoding import normalize_encoding

    enc = normalize_encoding(encoding, default="real")
    if enc == "permutation":
        return ("ox", {}), ("swap", {"prob": 0.1})
    if enc == "binary":
        return ("uniform", {"prob": 0.9}), ("bitflip", {"prob": mut_prob})
    if enc == "integer":
        return ("sbx", {"prob": 0.9, "eta": 20.0}), ("polynomial", {"prob": mut_prob, "eta": 20.0})
    if enc == "mixed":
        return ("mixed", {"prob": 0.9}), ("mixed", {"prob": mut_prob})
    # real (default)
    return ("sbx", {"prob": 1.0, "eta": 20.0}), ("polynomial", {"prob": mut_prob, "eta": 20.0})


def _require_fields(cfg: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    """Validate that required fields are present in configuration."""
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")


def _normalize_tournament_selection_kwargs(method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize tournament selection kwargs to use ``size`` as the public key.

    Backward compatibility: ``pressure`` is accepted as a legacy alias.
    """
    if str(method).strip().lower() != "tournament":
        return kwargs

    has_size = "size" in kwargs and kwargs.get("size") is not None
    has_pressure = "pressure" in kwargs and kwargs.get("pressure") is not None
    if has_size and has_pressure:
        raise ValueError("Tournament selection accepts either 'size' or legacy 'pressure', not both.")

    normalized = dict(kwargs)
    if has_pressure and not has_size:
        normalized["size"] = normalized.pop("pressure")
    return normalized

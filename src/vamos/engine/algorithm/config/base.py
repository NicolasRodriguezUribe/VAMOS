"""Base utilities for algorithm configuration."""

from __future__ import annotations

import dataclasses
import difflib
import json
import sys
import warnings
from dataclasses import asdict
from typing import Any, Literal, TypeAlias, cast, overload

from vamos.engine.algorithm.config.types import CrossoverName, InitializerName, MutationName, RepairName, SelectionName
from vamos.engine.archive import ExternalArchiveConfig
from vamos.foundation.encoding import EncodingLike

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# ── Shared Literal type aliases ──────────────────────────────────────
ResultMode = Literal["non_dominated", "population"]
ConstraintModeStr = Literal["none", "feasibility", "penalty", "epsilon"]
LiveCallbackMode = Literal["nd_only", "all"]
IndicatorType = Literal["eps", "hypervolume"]
OperatorCategory: TypeAlias = Literal["crossover", "mutation", "selection", "repair", "initializer"]

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
        "pm",
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
        "wrap",
        "wrapping",
        "midpoint",
        "midpoint_base",
        "gradient",
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

_EXTERNAL_ARCHIVE_ALLOWED_KWARGS = {
    "pruning",
    "nondominated_only",
    "epsilon",
    "hv_ref_point",
    "hv_samples",
    "rng_seed",
    "objective_tolerance",
    "truncate_size",
    "deduplicate_in",
    "decision_tolerance",
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


class _ConfigBuilderState:
    """Shared mutable state for fluent algorithm config builders."""

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}


class _PopSizeBuilder:
    _cfg: dict[str, Any]

    def pop_size(self, value: int) -> Self:
        self._cfg["pop_size"] = value
        return self


class _CrossoverBuilder:
    _cfg: dict[str, Any]

    @overload
    def crossover(self, method: CrossoverName, **kwargs: Any) -> Self: ...

    @overload
    def crossover(self, method: str, **kwargs: Any) -> Self: ...

    def crossover(self, method: str, **kwargs: Any) -> Self:
        if not isinstance(method, str):
            raise TypeError("crossover() expects a method name followed by keyword arguments, e.g. crossover('sbx', prob=1.0, eta=20.0).")
        self._cfg["crossover"] = (method, kwargs)
        return self


class _MutationBuilder:
    _cfg: dict[str, Any]

    @overload
    def mutation(self, method: MutationName, **kwargs: Any) -> Self: ...

    @overload
    def mutation(self, method: str, **kwargs: Any) -> Self: ...

    def mutation(self, method: str, **kwargs: Any) -> Self:
        if not isinstance(method, str):
            raise TypeError("mutation() expects a method name followed by keyword arguments, e.g. mutation('pm', prob='1/n', eta=20.0).")
        self._cfg["mutation"] = (method, kwargs)
        return self


class _SelectionBuilder:
    _cfg: dict[str, Any]

    @overload
    def selection(self, method: SelectionName, **kwargs: Any) -> Self: ...

    @overload
    def selection(self, method: str, **kwargs: Any) -> Self: ...

    def selection(self, method: str, **kwargs: Any) -> Self:
        self._cfg["selection"] = (method, _normalize_tournament_selection_kwargs(method, kwargs))
        return self


class _RepairBuilder:
    _cfg: dict[str, Any]

    @overload
    def repair(self, method: RepairName, **kwargs: Any) -> Self: ...

    @overload
    def repair(self, method: str, **kwargs: Any) -> Self: ...

    def repair(self, method: str, **kwargs: Any) -> Self:
        self._cfg["repair"] = (method, kwargs)
        return self


class _InitializerBuilder:
    _cfg: dict[str, Any]

    @overload
    def initializer(self, method: InitializerName, **kwargs: Any) -> Self: ...

    @overload
    def initializer(self, method: str, **kwargs: Any) -> Self: ...

    def initializer(self, method: str, **kwargs: Any) -> Self:
        self._cfg["initializer"] = {"type": method, **kwargs}
        return self


class _MutationProbFactorBuilder:
    _cfg: dict[str, Any]

    def mutation_prob_factor(self, value: float) -> Self:
        self._cfg["mutation_prob_factor"] = float(value)
        return self


class _ConstraintModeBuilder:
    _cfg: dict[str, Any]

    def constraint_mode(self, value: ConstraintModeStr) -> Self:
        self._cfg["constraint_mode"] = value
        return self


class _TrackGenealogyBuilder:
    _cfg: dict[str, Any]

    def track_genealogy(self, enabled: bool = True) -> Self:
        self._cfg["track_genealogy"] = bool(enabled)
        return self


class _ResultArchiveBuilder:
    _cfg: dict[str, Any]

    def result_mode(self, value: ResultMode) -> Self:
        mode = str(value).strip().lower()
        if mode not in {"non_dominated", "population"}:
            raise ValueError("result_mode must be 'non_dominated' or 'population'.")
        self._cfg["result_mode"] = mode
        return self

    def external_archive(self, capacity: int | None = None, **kwargs: Any) -> Self:
        """Configure an external archive.

        Parameters
        ----------
        capacity
            Maximum number of solutions. ``None`` means unbounded.
        **kwargs
            Forwarded to :class:`ExternalArchiveConfig`.
        """
        self._cfg["external_archive"] = _build_external_archive_config(capacity, kwargs)
        self._cfg.setdefault("result_mode", "non_dominated")
        return self


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    __dataclass_fields__: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(cast(Any, self)))
        return _omit_inactive_archive_fields(payload)

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
    def available_operators(category: OperatorCategory | None = None) -> dict[str, list[str]]:
        """Return known operator names, optionally filtered by *category*.

        Parameters
        ----------
        category : OperatorCategory | None, optional
            Requested operator category. ``None`` returns all categories.
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
    def default_operators(encoding: EncodingLike = "real") -> dict[str, tuple[str, dict[str, Any]]]:
        """Return the default crossover/mutation operators for *encoding*.

        >>> NSGAIIConfig.default_operators("permutation")
        {'crossover': ('ox', {}), 'mutation': ('swap', {'prob': 0.1})}
        """
        cx, mt = _default_operators_for_encoding(encoding, mut_prob=0.1)
        return {"crossover": cx, "mutation": mt}


def _default_operators_for_encoding(
    encoding: EncodingLike,
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


def _build_external_archive_config(capacity: int | None, kwargs: dict[str, Any]) -> ExternalArchiveConfig:
    unexpected = sorted(set(kwargs) - _EXTERNAL_ARCHIVE_ALLOWED_KWARGS)
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"external_archive() got unexpected keyword argument(s): {names}")
    return ExternalArchiveConfig(capacity=capacity, **kwargs)


def _omit_inactive_archive_fields(payload: dict[str, object]) -> dict[str, object]:
    external_archive = payload.get("external_archive")
    if isinstance(external_archive, dict) and external_archive.get("capacity") is None:
        external_archive.pop("pruning", None)
    return payload


def _normalize_tournament_selection_kwargs(method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate tournament selection kwargs to use ``size`` as the public key."""
    if str(method).strip().lower() != "tournament":
        return kwargs

    if "pressure" in kwargs and kwargs.get("pressure") is not None:
        raise ValueError("Tournament selection uses 'size'; 'pressure' is no longer supported.")
    return dict(kwargs)

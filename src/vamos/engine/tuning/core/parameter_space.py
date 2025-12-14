from __future__ import annotations

"""
Canonical tuning configuration space for VAMOS.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


class ParameterType:
    """Base class for tunable parameter domains."""

    def decode(self, u: float) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _clip_unit(u: float) -> float:
        if not np.isfinite(u):
            raise ValueError("Parameter decoding received a non-finite value.")
        if u < 0.0 or u > 1.0:
            raise ValueError("Parameter decoding expects values within [0, 1].")
        return float(u)


@dataclass(frozen=True)
class Categorical(ParameterType):
    values: Sequence[str]

    def decode(self, u: float) -> str:
        u = self._clip_unit(u)
        if not self.values:
            raise ValueError("Categorical parameter requires at least one option.")
        idx = min(int(np.floor(u * len(self.values))), len(self.values) - 1)
        return str(self.values[idx])


@dataclass(frozen=True)
class CategoricalInteger(ParameterType):
    values: Sequence[int]

    def decode(self, u: float) -> int:
        u = self._clip_unit(u)
        if not self.values:
            raise ValueError("CategoricalInteger parameter requires at least one option.")
        idx = min(int(np.floor(u * len(self.values))), len(self.values) - 1)
        return int(self.values[idx])


@dataclass(frozen=True)
class Integer(ParameterType):
    low: int
    high: int

    def decode(self, u: float) -> int:
        u = self._clip_unit(u)
        if self.high < self.low:
            raise ValueError("Integer parameter requires high >= low.")
        span = self.high - self.low
        raw = int(self.low + np.floor(u * (span + 1)))
        return min(self.high, max(self.low, raw))


@dataclass(frozen=True)
class Double(ParameterType):
    low: float
    high: float

    def decode(self, u: float) -> float:
        u = self._clip_unit(u)
        if self.high < self.low:
            raise ValueError("Double parameter requires high >= low.")
        return float(self.low + u * (self.high - self.low))


@dataclass(frozen=True)
class Boolean(ParameterType):
    def decode(self, u: float) -> bool:
        u = self._clip_unit(u)
        return bool(u >= 0.5)


@dataclass
class ParameterDefinition:
    param: ParameterType
    sub_parameters: Dict[str, "ParameterDefinition"] = field(default_factory=dict)
    conditional_sub_parameters: Dict[Any, Dict[str, "ParameterDefinition"]] = field(default_factory=dict)
    fixed_sub_parameters: Dict[str, Any] = field(default_factory=dict)

    def dim(self) -> int:
        total = 1
        for child in self.sub_parameters.values():
            total += child.dim()
        for branch in self.conditional_sub_parameters.values():
            for child in branch.values():
                total += child.dim()
        return total

    def decode(self, vector: np.ndarray, start: int) -> tuple[Any, int]:
        if start >= vector.size:
            raise ValueError("Parameter decoding received an incomplete vector.")
        value = self.param.decode(float(vector[start]))
        idx = start + 1

        params: Dict[str, Any] = dict(self.fixed_sub_parameters)
        for name, child in self.sub_parameters.items():
            decoded, idx = child.decode(vector, idx)
            params[name] = decoded

        for activation_value, branch in self.conditional_sub_parameters.items():
            active = activation_value == value
            for name, child in branch.items():
                decoded, idx = child.decode(vector, idx)
                if active:
                    params[name] = decoded

        if params:
            return (value, params), idx
        return value, idx

    def flatten_names(self, base_name: str) -> list[str]:
        names = [base_name]
        for name, child in self.sub_parameters.items():
            names.extend(child.flatten_names(f"{base_name}.{name}"))
        for activation_value, branch in self.conditional_sub_parameters.items():
            for name, child in branch.items():
                names.extend(child.flatten_names(f"{base_name}[{activation_value!r}].{name}"))
        return names


class AlgorithmConfigSpace:
    """
    Flattened view over an AlgorithmConfig builder with hierarchical parameters.
    Supports individual algorithm spaces or a multi-algorithm categorical wrapper.
    """

    def __init__(
        self,
        config_builder=None,
        parameters: Mapping[str, ParameterDefinition] | None = None,
        fixed_values: Mapping[str, Any] | None = None,
        setter_overrides: Mapping[str, str] | None = None,
        *,
        algorithms: Mapping[str, "AlgorithmConfigSpace"] | None = None,
        template_name: str | None = None,
    ):
        if algorithms is not None and config_builder is not None:
            raise ValueError("Provide either a single config_builder or multi-algorithm mapping, not both.")
        self._mode_multi = algorithms is not None
        self._template_name = template_name
        if self._mode_multi:
            if not algorithms:
                raise ValueError("Multi-algorithm space requires at least one algorithm space.")
            self._algo_spaces: Dict[str, AlgorithmConfigSpace] = dict(algorithms)
            self._algo_names: list[str] = list(self._algo_spaces.keys())
            self._algo_choice = ParameterDefinition(Categorical(self._algo_names))
            self._prefix_dims: Dict[str, int] = {}
            dim = 1
            for name in self._algo_names:
                dim += self._algo_spaces[name].dim()
            self._dim = dim
        else:
            if not callable(config_builder):
                raise TypeError("config_builder must be callable and return a builder instance.")
            if parameters is None:
                parameters = {}
            self._builder_factory = config_builder
            self._parameters = list(parameters.items())
            self._fixed = dict(fixed_values or {})
            self._setter_overrides = dict(setter_overrides or {})
            self._dim = sum(param.dim() for _, param in self._parameters)

    @classmethod
    def from_template(cls, algorithm: str, template_name: str) -> "AlgorithmConfigSpace":
        from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SPEA2Config, IBEAConfig, SMPSOConfig

        template = template_name.lower()
        algo = algorithm.lower()
        if algo == "nsgaii":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "crossover": ParameterDefinition(
                    Categorical(["sbx"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.7, 1.0)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "selection": ParameterDefinition(
                    Categorical(["tournament"]),
                    sub_parameters={"pressure": ParameterDefinition(CategoricalInteger([2, 3, 4]))},
                ),
            }
            fixed = {"survival": "nsga2", "engine": "numpy"}
            return cls(NSGAIIConfig, params, fixed, template_name=template)

        if algo == "moead":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "neighbor_size": ParameterDefinition(Integer(5, 40)),
                "delta": ParameterDefinition(Double(0.5, 0.99)),
                "replace_limit": ParameterDefinition(Integer(1, 5)),
                "crossover": ParameterDefinition(
                    Categorical(["sbx"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.7, 1.0)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "aggregation": ParameterDefinition(Categorical(["tchebycheff"])),
            }
            fixed = {"engine": "numpy", "weight_vectors": {"path": None, "divisions": None}}
            return cls(MOEADConfig, params, fixed, template_name=template)

        if algo == "nsga3":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "crossover": ParameterDefinition(
                    Categorical(["sbx"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.7, 1.0)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "selection": ParameterDefinition(
                    Categorical(["tournament"]),
                    sub_parameters={"pressure": ParameterDefinition(CategoricalInteger([2, 3, 4]))},
                ),
                "reference_directions": ParameterDefinition(
                    Categorical(["precomputed"]),
                    sub_parameters={"divisions": ParameterDefinition(Integer(10, 30))},
                ),
            }
            fixed = {"engine": "numpy"}
            return cls(NSGAIIIConfig, params, fixed, template_name=template)

        if algo == "spea2":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "archive_size": ParameterDefinition(Integer(50, 300)),
                "crossover": ParameterDefinition(
                    Categorical(["sbx"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.7, 1.0)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "selection": ParameterDefinition(
                    Categorical(["tournament"]),
                    sub_parameters={"pressure": ParameterDefinition(CategoricalInteger([2, 3, 4]))},
                ),
                "k_neighbors": ParameterDefinition(Integer(1, 25)),
            }
            fixed = {"engine": "numpy"}
            return cls(SPEA2Config, params, fixed, template_name=template)

        if algo == "ibea":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "crossover": ParameterDefinition(
                    Categorical(["sbx"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.7, 1.0)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
                "selection": ParameterDefinition(
                    Categorical(["tournament"]),
                    sub_parameters={"pressure": ParameterDefinition(CategoricalInteger([2, 3, 4]))},
                ),
                "indicator": ParameterDefinition(Categorical(["eps", "hypervolume"])),
                "kappa": ParameterDefinition(Double(0.01, 0.2)),
            }
            fixed = {"engine": "numpy"}
            return cls(IBEAConfig, params, fixed, template_name=template)

        if algo == "smpso":
            params = {
                "pop_size": ParameterDefinition(Integer(50, 300)),
                "archive_size": ParameterDefinition(Integer(50, 300)),
                "inertia": ParameterDefinition(Double(0.1, 0.9)),
                "c1": ParameterDefinition(Double(0.5, 2.5)),
                "c2": ParameterDefinition(Double(0.5, 2.5)),
                "vmax_fraction": ParameterDefinition(Double(0.1, 1.0)),
                "mutation": ParameterDefinition(
                    Categorical(["pm"]),
                    sub_parameters={
                        "prob": ParameterDefinition(Double(0.01, 0.5)),
                        "eta": ParameterDefinition(Double(5.0, 40.0)),
                    },
                ),
            }
            fixed = {"engine": "numpy"}
            return cls(SMPSOConfig, params, fixed, template_name=template)

        raise ValueError(f"Unsupported template '{template_name}' for algorithm '{algorithm}'.")

    @classmethod
    def multi_algorithm(cls, spaces: Mapping[str, "AlgorithmConfigSpace"]) -> "AlgorithmConfigSpace":
        return cls(algorithms=spaces)

    def dim(self) -> int:
        return self._dim

    def parameter_names(self) -> list[str]:
        if self._mode_multi:
            names = self._algo_choice.flatten_names("algorithm")
            for name in self._algo_names:
                subspace = self._algo_spaces[name]
                prefix = f"{name}"
                names.extend([f"{prefix}.{n}" for n in subspace.parameter_names()])
            return names
        names: list[str] = []
        for name, param in self._parameters:
            names.extend(param.flatten_names(name))
        return names

    def decode_vector(self, x: np.ndarray):
        vector = np.asarray(x, dtype=float)
        if vector.ndim != 1:
            raise ValueError("Meta-configuration vector must be one-dimensional.")
        if vector.size != self._dim:
            raise ValueError(f"Expected vector of length {self._dim}, received {vector.size}.")
        if np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("Configuration vectors must lie within the unit hypercube.")

        if self._mode_multi:
            return self._decode_multi(vector)

        decoded: Dict[str, Any] = {}
        idx = 0
        for name, param in self._parameters:
            value, idx = param.decode(vector, idx)
            decoded[name] = value
        if idx != vector.size:
            raise ValueError("Parameter decoding did not consume the entire vector.")

        decoded.update(self._fixed)
        builder = self._builder_factory()
        apply = self._apply_to_builder
        for field_name, value in decoded.items():
            apply(builder, field_name, value)
        if hasattr(builder, "fixed"):
            return builder.fixed()
        if hasattr(builder, "freeze"):
            return builder.freeze()
        return builder

    def _decode_multi(self, vector: np.ndarray):
        algo_name, idx = self._algo_choice.decode(vector, 0)
        algo_name = str(algo_name)
        configs: Dict[str, Any] = {}
        for name in self._algo_names:
            subspace = self._algo_spaces[name]
            sub_dim = subspace.dim()
            segment = vector[idx : idx + sub_dim]
            if segment.size != sub_dim:
                raise ValueError("Incomplete meta-vector for algorithm sub-space decoding.")
            configs[name] = subspace.decode_vector(segment)
            idx += sub_dim
        if idx != vector.size:
            raise ValueError("Multi-algorithm decoding did not consume the entire vector.")
        if algo_name not in configs:
            raise ValueError(f"Decoded algorithm '{algo_name}' not present in sub-spaces.")
        return configs[algo_name]

    def _apply_to_builder(self, builder, field_name: str, value: Any) -> None:
        setter_name = getattr(self, "_setter_overrides", {}).get(field_name, field_name)
        setter = getattr(builder, setter_name, None)
        if setter is None or not callable(setter):
            raise AttributeError(f"Builder missing setter for field '{field_name}'.")
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            setter(value[0], **value[1])
        elif isinstance(value, dict):
            setter(**value)
        else:
            setter(value)

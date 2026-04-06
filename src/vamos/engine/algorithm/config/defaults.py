"""Default config builders for built-in algorithms."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from vamos.foundation.encoding import normalize_encoding

from .agemoea import AGEMOEAConfig
from .base import _default_operators_for_encoding
from .ibea import IBEAConfig
from .moead import MOEADConfig
from .nsgaii import NSGAIIConfig
from .nsgaiii import NSGAIIIConfig
from .rvea import RVEAConfig
from .smpso import SMPSOConfig
from .smsemoa import SMSEMOAConfig
from .spea2 import SPEA2Config
from .types import AlgorithmConfigProtocol

DefaultConfigBuilder = Callable[[int | None, int | None, int | None, str | None], AlgorithmConfigProtocol]


def _mutation_prob(n_var: int | None) -> float:
    return 1.0 / n_var if n_var else 0.1


def _call_default(default_fn: Callable[..., AlgorithmConfigProtocol], **kwargs: int | str | None) -> AlgorithmConfigProtocol:
    accepted = inspect.signature(default_fn).parameters
    return default_fn(**{key: value for key, value in kwargs.items() if key in accepted})


def _nsgaii_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(NSGAIIConfig.default, pop_size=pop_size, n_var=n_var, encoding=encoding)


def _moead_default(
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    n_obj_value = 3 if n_obj is None else n_obj
    normalized = normalize_encoding(encoding, default="real")
    if normalized == "real":
        return _call_default(MOEADConfig.default, pop_size=pop_size, n_var=n_var, n_obj=n_obj_value)

    cx, mt = _default_operators_for_encoding(normalized, _mutation_prob(n_var))
    builder = MOEADConfig.builder()
    builder.pop_size(100 if pop_size is None else pop_size)
    builder.neighbor_size(20)
    builder.delta(0.9)
    builder.replace_limit(2)
    builder.crossover(cx[0], **cx[1])
    builder.mutation(mt[0], **mt[1])
    builder.aggregation("pbi", theta=5.0)
    return builder.build()


def _smsemoa_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(SMSEMOAConfig.default, pop_size=pop_size, n_var=n_var)


def _nsgaiii_default(
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    n_obj_value = 3 if n_obj is None else n_obj
    return _call_default(NSGAIIIConfig.default, pop_size=pop_size, n_var=n_var, n_obj=n_obj_value)


def _spea2_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(SPEA2Config.default, pop_size=100 if pop_size is None else pop_size, n_var=n_var)


def _ibea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(IBEAConfig.default, pop_size=100 if pop_size is None else pop_size, n_var=n_var)


def _smpso_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(SMPSOConfig.default, pop_size=100 if pop_size is None else pop_size, n_var=n_var)


def _agemoea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(AGEMOEAConfig.default, pop_size=100 if pop_size is None else pop_size, n_var=n_var)


def _rvea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    return _call_default(RVEAConfig.default, pop_size=100 if pop_size is None else pop_size, n_var=n_var)


_DEFAULT_BUILDERS: dict[str, DefaultConfigBuilder] = {
    "nsgaii": _nsgaii_default,
    "gces": _nsgaii_default,
    "gces_nocomp": _nsgaii_default,
    "gces_nogeo": _nsgaii_default,
    "nsga2_farthest": _nsgaii_default,
    "nsga2_gapfill": _nsgaii_default,
    "nsga2_curvgap": _nsgaii_default,
    "nsga2_hvfarthest": _nsgaii_default,
    "nsga2_refcover_farthest": _nsgaii_default,
    "nsga2_hvref_farthest": _nsgaii_default,
    "nsga2_sector_farthest": _nsgaii_default,
    "moead": _moead_default,
    "smsemoa": _smsemoa_default,
    "nsgaiii": _nsgaiii_default,
    "spea2": _spea2_default,
    "ibea": _ibea_default,
    "smpso": _smpso_default,
    "agemoea": _agemoea_default,
    "rvea": _rvea_default,
}


def build_default_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
) -> AlgorithmConfigProtocol | None:
    builder = _DEFAULT_BUILDERS.get(algorithm.lower())
    if builder is None:
        return None
    return builder(pop_size, n_var, n_obj, encoding)


__all__ = ["build_default_algorithm_config", "DefaultConfigBuilder"]

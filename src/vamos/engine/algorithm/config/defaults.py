"""Default config builders for built-in algorithms."""

from __future__ import annotations

from collections.abc import Callable

from .agemoea import AGEMOEAConfig
from .ibea import IBEAConfig
from .moead import MOEADConfig
from .nsgaii import NSGAIIConfig
from .nsgaiii import NSGAIIIConfig
from .rvea import RVEAConfig
from .smsemoa import SMSEMOAConfig
from .smpso import SMPSOConfig
from .spea2 import SPEA2Config
from .types import AlgorithmConfigProtocol


DefaultConfigBuilder = Callable[[int | None, int | None, int | None, str | None], AlgorithmConfigProtocol]


def _nsgaii_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return NSGAIIConfig.default(n_var=n_var, encoding=encoding, **kwargs)


def _moead_default(
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    n_obj_value = 3 if n_obj is None else n_obj
    return MOEADConfig.default(pop_size=pop_size, n_var=n_var, n_obj=n_obj_value)


def _smsemoa_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return SMSEMOAConfig.default(n_var=n_var, **kwargs)


def _nsgaiii_default(
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    n_obj_value = 3 if n_obj is None else n_obj
    return NSGAIIIConfig.default(pop_size=pop_size, n_var=n_var, n_obj=n_obj_value)


def _spea2_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return SPEA2Config.default(n_var=n_var, **kwargs)


def _ibea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return IBEAConfig.default(n_var=n_var, **kwargs)


def _smpso_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return SMPSOConfig.default(n_var=n_var, **kwargs)


def _agemoea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return AGEMOEAConfig.default(n_var=n_var, **kwargs)


def _rvea_default(
    pop_size: int | None,
    n_var: int | None,
    _n_obj: int | None,
    _encoding: str | None,
) -> AlgorithmConfigProtocol:
    kwargs: dict[str, int] = {}
    if pop_size is not None:
        kwargs["pop_size"] = pop_size
    return RVEAConfig.default(n_var=n_var, **kwargs)


_DEFAULT_BUILDERS: dict[str, DefaultConfigBuilder] = {
    "nsgaii": _nsgaii_default,
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

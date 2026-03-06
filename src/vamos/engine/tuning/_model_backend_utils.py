from __future__ import annotations

from typing import Any

import numpy as np

from .racing.param_space import Boolean, Categorical, Int, ParamSpace, Real

_OPTUNA_SAMPLERS: dict[str, str] = {
    "tpe": "TPESampler",
    "cmaes": "CmaEsSampler",
    "random": "RandomSampler",
    "nsgaii": "NSGAIISampler",
    "nsgaiii": "NSGAIIISampler",
    "qmc": "QMCSampler",
    "gp": "GPSampler",
}


def sample_from_optuna_trial(trial: Any, param_space: ParamSpace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for name, spec in param_space.params.items():
        if isinstance(spec, Real):
            cfg[name] = float(trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Int):
            cfg[name] = int(trial.suggest_int(name, int(spec.low), int(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Categorical):
            cfg[name] = trial.suggest_categorical(name, list(spec.choices))
        elif isinstance(spec, Boolean):
            cfg[name] = bool(trial.suggest_categorical(name, [False, True]))
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
    return cfg


def build_configspace(param_space: ParamSpace, seed: int) -> Any:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    cs = ConfigurationSpace(seed=int(seed))
    for name, spec in param_space.params.items():
        hp: Any
        if isinstance(spec, Real):
            hp = UniformFloatHyperparameter(name=name, lower=float(spec.low), upper=float(spec.high), log=bool(spec.log))
        elif isinstance(spec, Int):
            hp = UniformIntegerHyperparameter(name=name, lower=int(spec.low), upper=int(spec.high), log=bool(spec.log))
        elif isinstance(spec, Categorical):
            hp = CategoricalHyperparameter(name=name, choices=list(spec.choices))
        elif isinstance(spec, Boolean):
            hp = CategoricalHyperparameter(name=name, choices=[False, True])
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
        cs.add_hyperparameter(hp)
    return cs


def estimate_hyperband_evals_per_iteration(max_budget: int, eta: int) -> int:
    max_budget = max(1, int(max_budget))
    eta = max(2, int(eta))
    if max_budget <= 1:
        return 1
    min_budget = 1.0
    s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))
    B = (s_max + 1) * max_budget
    total = 0
    for s in range(s_max, -1, -1):
        n = int(np.ceil((B / max_budget / (s + 1)) * (eta**s)))
        n = max(1, n)
        for i in range(s + 1):
            n_i = int(np.floor(n * (eta ** (-i))))
            total += max(1, n_i)
    return max(1, total)


def build_optuna_sampler(name: str, seed: int) -> Any:
    import optuna.samplers as samplers

    key = name.lower().replace("-", "").replace("_", "")
    cls_name = _OPTUNA_SAMPLERS.get(key)
    if cls_name is None:
        supported = ", ".join(sorted(_OPTUNA_SAMPLERS))
        raise ValueError(f"Unknown optuna_sampler {name!r}. Choose from: {supported}")
    cls = getattr(samplers, cls_name)
    if key == "tpe":
        return cls(seed=seed, multivariate=True, group=True)
    return cls(seed=seed)

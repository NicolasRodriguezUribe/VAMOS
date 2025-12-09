from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Sequence

import numpy as np

from .param_space import ParamSpace, Real, Int, Categorical


class Sampler(Protocol):
    """
    Interface for configuration samplers used by tuners.

    A sampler knows how to draw new configurations (dicts) from a given
    ParamSpace, possibly using a learned model of 'good regions' in the
    hyperparameter space.
    """

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a new configuration.

        Args:
            rng: NumPy random generator to use.

        Returns:
            A configuration dictionary mapping parameter names to values.
        """
        ...


@dataclass
class UniformSampler:
    """
    Uniform sampler that delegates to ParamSpace.sample.

    This reproduces the current behavior of the base Tuner and RacingTuner.
    """

    param_space: ParamSpace

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        return self.param_space.sample(rng)


@dataclass
class ModelBasedSampler:
    """
    Simple model-based sampler that learns per-parameter marginal distributions
    from a set of 'good' configurations.

    - For categorical parameters, it estimates empirical frequencies.
    - For real and integer parameters, it shrinks the sampling interval around
      the min/max of observed good values (clipped to original bounds).

    Sampling uses a mixture of:
        - exploration: uniform sampling from the full ParamSpace,
        - exploitation: sampling from learned marginals.
    """

    param_space: ParamSpace
    exploration_prob: float = 0.2
    """
    Probability of ignoring the learned model and sampling uniformly from the
    ParamSpace. This ensures persistent exploration.
    """

    min_samples_to_model: int = 5
    """
    Minimum number of 'good' configurations required before using the learned
    marginal model for exploitation sampling.
    """

    _cat_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _real_models: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _int_models: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def update(self, good_configs: Sequence[Mapping[str, Any]]) -> None:
        """
        Update the internal marginal model from a collection of 'good' configs.

        If the number of good_configs is below min_samples_to_model, the internal
        model is cleared and only uniform sampling will be used.
        """
        if len(good_configs) < self.min_samples_to_model:
            self._cat_models.clear()
            self._real_models.clear()
            self._int_models.clear()
            return

        observed: Dict[str, List[Any]] = {name: [] for name in self.param_space.params.keys()}

        for cfg in good_configs:
            for name in observed.keys():
                if name in cfg:
                    observed[name].append(cfg[name])

        self._cat_models.clear()
        self._real_models.clear()
        self._int_models.clear()

        for name, spec in self.param_space.params.items():
            values = observed.get(name, [])
            if not values:
                continue

            if isinstance(spec, Categorical):
                counts: Dict[Any, int] = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1

                total = sum(counts.values())
                if total == 0:
                    continue

                choices: List[Any] = []
                probs: List[float] = []
                for choice in spec.choices:
                    c = counts.get(choice, 0)
                    if c > 0:
                        choices.append(choice)
                        probs.append(c / total)

                if not choices:
                    continue

                self._cat_models[name] = {
                    "choices": choices,
                    "probs": np.asarray(probs, dtype=float),
                }

            elif isinstance(spec, Real):
                v_min = float(min(values))
                v_max = float(max(values))
                orig_low = spec.low
                orig_high = spec.high

                padding = 0.05 * (orig_high - orig_low)
                low = max(orig_low, v_min - padding)
                high = min(orig_high, v_max + padding)

                if low > high:
                    low, high = high, low

                self._real_models[name] = {"low": low, "high": high}

            elif isinstance(spec, Int):
                v_min = int(min(values))
                v_max = int(max(values))
                orig_low = spec.low
                orig_high = spec.high

                padding = max(1, int(round(0.05 * (orig_high - orig_low))))
                low = max(orig_low, v_min - padding)
                high = min(orig_high, v_max + padding)

                if low > high:
                    low, high = high, low

                self._int_models[name] = {"low": low, "high": high}

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a new configuration using a mixture of uniform exploration and
        marginal exploitation. If insufficient model data exist, falls back to
        uniform sampling.
        """
        if rng.random() < self.exploration_prob or (
            not self._cat_models and not self._real_models and not self._int_models
        ):
            return self.param_space.sample(rng)

        cfg: Dict[str, Any] = {}

        for name, spec in self.param_space.params.items():
            if isinstance(spec, Categorical) and name in self._cat_models:
                model = self._cat_models[name]
                choices = model["choices"]
                probs = model["probs"]
                idx = int(rng.choice(len(choices), p=probs))
                cfg[name] = choices[idx]
                continue

            if isinstance(spec, Real) and name in self._real_models:
                m = self._real_models[name]
                cfg[name] = float(rng.uniform(m["low"], m["high"]))
                continue

            if isinstance(spec, Int) and name in self._int_models:
                m = self._int_models[name]
                cfg[name] = int(rng.integers(m["low"], m["high"] + 1))
                continue

            tmp = self.param_space.sample(rng)
            cfg[name] = tmp[name]

        return cfg


__all__ = ["Sampler", "UniformSampler", "ModelBasedSampler"]

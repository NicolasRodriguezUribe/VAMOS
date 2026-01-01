from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import moocore as _moocore  # type: ignore

    _MC = _moocore._moocore
except Exception:  # pragma: no cover - optional dependency
    _moocore = None
    _MC = None


def has_moocore() -> bool:
    return _MC is not None


@dataclass
class IndicatorResult:
    value: float | np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


class QualityIndicator(Protocol):
    name: str

    def compute(
        self,
        front: np.ndarray,
        reference_front: Optional[np.ndarray] = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult: ...


def _require_moocore() -> None:
    if not has_moocore():
        raise ImportError("MooCore is required for this indicator but is not installed.")


def _to_minimization(front: np.ndarray, maximise: bool | Sequence[bool] | None) -> np.ndarray:
    if maximise is None:
        return front
    arr = np.asarray(front, dtype=float)
    if isinstance(maximise, bool):
        return -arr if maximise else arr
    mask = np.asarray(maximise, dtype=bool)
    flipped = arr.copy()
    flipped[:, mask] = -flipped[:, mask]
    return flipped


@dataclass
class HVIndicator(QualityIndicator):
    reference_point: Optional[np.ndarray] = None
    name: str = "hv"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        ref = np.asarray(self.reference_point if self.reference_point is not None else np.max(F, axis=0) + 1.0, dtype=float)
        val = float(_MC.hypervolume(F, ref=ref))
        return IndicatorResult(value=val, details={"reference_point": ref})


@dataclass
class HVContributionsIndicator(QualityIndicator):
    reference_point: Optional[np.ndarray] = None
    name: str = "hv_contributions"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        ref = np.asarray(self.reference_point if self.reference_point is not None else np.max(F, axis=0) + 1.0, dtype=float)
        contrib = np.asarray(_MC.hv_contributions(F, ref=ref), dtype=float)
        return IndicatorResult(value=contrib, details={"reference_point": ref})


@dataclass
class IGDIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "igd"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        val = float(_MC.igd(ref, F))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class IGDPlusIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "igd_plus"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        val = float(_MC.igd_plus(ref, F))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class EpsilonAdditiveIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "epsilon_additive"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        val = float(_MC.epsilon_additive(ref, F))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class EpsilonMultiplicativeIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "epsilon_multiplicative"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **_: Any) -> IndicatorResult:
        _require_moocore()
        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        val = float(_MC.epsilon_mult(ref, F))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class AvgHausdorffIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "avg_hausdorff"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **kwargs: Any) -> IndicatorResult:
        _require_moocore()
        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        p = kwargs.get("p", 1.0)
        val = float(_MC.avg_hausdorff_dist(ref, F, p=p))
        return IndicatorResult(value=val, details={"reference_front": ref, "p": p})


@dataclass
class HVApproxIndicator(QualityIndicator):
    reference_point: Optional[np.ndarray] = None
    name: str = "hv_approx"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **kwargs: Any) -> IndicatorResult:
        _require_moocore()
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        ref = np.asarray(self.reference_point if self.reference_point is not None else np.max(F, axis=0) + 1.0, dtype=float)
        samples = int(kwargs.get("samples", 10000))
        val = float(_MC.hv_approx(F, ref=ref, maximise=False, nsamples=samples))
        return IndicatorResult(value=val, details={"reference_point": ref, "samples": samples})


@dataclass
class WHVRectIndicator(QualityIndicator):
    reference_point: Optional[np.ndarray] = None
    rectangles: Optional[np.ndarray] = None
    name: str = "whv_rect"

    def compute(self, front: np.ndarray, reference_front: Optional[np.ndarray] = None, maximise=None, **kwargs: Any) -> IndicatorResult:
        _require_moocore()
        F = _to_minimization(np.asarray(front, dtype=float), maximise)
        ref = np.asarray(self.reference_point if self.reference_point is not None else np.max(F, axis=0) + 1.0, dtype=float)
        rects = np.asarray(self.rectangles if self.rectangles is not None else kwargs.get("rectangles"), dtype=float)
        if rects is None or rects.size == 0:
            raise ValueError("whv_rect requires rectangles to be provided.")
        val = float(_MC.whv_rect(F, rects, ref=ref, maximise=False))
        return IndicatorResult(value=val, details={"reference_point": ref})


def get_indicator(name: str, **kwargs: Any) -> QualityIndicator:
    key = name.lower()
    if key in {"hv", "hypervolume"}:
        return HVIndicator(**kwargs)
    if key in {"hv_contrib", "hv_contributions"}:
        return HVContributionsIndicator(**kwargs)
    if key == "igd":
        return IGDIndicator(**kwargs)
    if key in {"igd+", "igd_plus"}:
        return IGDPlusIndicator(**kwargs)
    if key in {"eps", "epsilon", "epsilon_add", "epsilon_additive"}:
        return EpsilonAdditiveIndicator(**kwargs)
    if key in {"epsilon_mult", "eps_mult"}:
        return EpsilonMultiplicativeIndicator(**kwargs)
    if key in {"avg_hausdorff", "hausdorff"}:
        return AvgHausdorffIndicator(**kwargs)
    if key == "hv_approx":
        return HVApproxIndicator(**kwargs)
    if key in {"whv_rect", "weighted_hv"}:
        return WHVRectIndicator(**kwargs)
    raise ValueError(f"Unknown indicator '{name}'.")


__all__ = [
    "IndicatorResult",
    "QualityIndicator",
    "has_moocore",
    "get_indicator",
    "HVIndicator",
    "HVContributionsIndicator",
    "IGDIndicator",
    "IGDPlusIndicator",
    "EpsilonAdditiveIndicator",
    "EpsilonMultiplicativeIndicator",
    "AvgHausdorffIndicator",
    "HVApproxIndicator",
    "WHVRectIndicator",
]

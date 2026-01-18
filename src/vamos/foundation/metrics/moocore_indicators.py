from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except Exception:  # pragma: no cover - optional dependency
    _moocore = None


def has_moocore() -> bool:
    return _moocore is not None


@dataclass
class IndicatorResult:
    value: float | np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


class QualityIndicator(Protocol):
    name: str

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult: ...


def _require_moocore() -> None:
    if not has_moocore():
        raise ImportError("MooCore is required for this indicator but is not installed.")


def _normalize_maximise(maximise: bool | Sequence[bool] | None) -> bool | Sequence[bool]:
    return False if maximise is None else maximise


def _maximise_mask(maximise: bool | Sequence[bool], n_obj: int) -> np.ndarray:
    if isinstance(maximise, bool):
        return np.full(n_obj, maximise, dtype=bool)
    mask = np.asarray(maximise, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != n_obj:
        raise ValueError("maximise must be a bool or a sequence with one entry per objective.")
    return mask


def _default_reference_point(front: np.ndarray, maximise: bool | Sequence[bool]) -> np.ndarray:
    arr = np.asarray(front, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("front must be a non-empty 2D array.")
    n_obj = arr.shape[1]
    mask = _maximise_mask(maximise, n_obj)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    ref = np.empty(n_obj, dtype=float)
    ref[mask] = mins[mask] - 1.0
    ref[~mask] = maxs[~mask] + 1.0
    return ref


@dataclass
class HVIndicator(QualityIndicator):
    reference_point: np.ndarray | None = None
    name: str = "hv"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        ref = np.asarray(
            self.reference_point if self.reference_point is not None else _default_reference_point(F, maximise_flag),
            dtype=float,
        )
        val = float(moocore.hypervolume(F, ref=ref, maximise=maximise_flag))
        return IndicatorResult(value=val, details={"reference_point": ref})


@dataclass
class HVContributionsIndicator(QualityIndicator):
    reference_point: np.ndarray | None = None
    name: str = "hv_contributions"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        ref = np.asarray(
            self.reference_point if self.reference_point is not None else _default_reference_point(F, maximise_flag),
            dtype=float,
        )
        contrib = np.asarray(moocore.hv_contributions(F, ref=ref, maximise=maximise_flag), dtype=float)
        return IndicatorResult(value=contrib, details={"reference_point": ref})


@dataclass
class IGDIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "igd"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        val = float(moocore.igd(F, ref=ref, maximise=maximise_flag))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class IGDPlusIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "igd_plus"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        val = float(moocore.igd_plus(F, ref=ref, maximise=maximise_flag))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class EpsilonAdditiveIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "epsilon_additive"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        val = float(moocore.epsilon_additive(F, ref=ref, maximise=maximise_flag))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class EpsilonMultiplicativeIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "epsilon_multiplicative"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **_: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        val = float(moocore.epsilon_mult(F, ref=ref, maximise=maximise_flag))
        return IndicatorResult(value=val, details={"reference_front": ref})


@dataclass
class AvgHausdorffIndicator(QualityIndicator):
    reference_front: np.ndarray
    name: str = "avg_hausdorff"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        ref = np.asarray(reference_front if reference_front is not None else self.reference_front, dtype=float)
        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        p = kwargs.get("p", 1.0)
        val = float(moocore.avg_hausdorff_dist(F, ref, maximise=maximise_flag, p=p))
        return IndicatorResult(value=val, details={"reference_front": ref, "p": p})


@dataclass
class HVApproxIndicator(QualityIndicator):
    reference_point: np.ndarray | None = None
    name: str = "hv_approx"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        ref = np.asarray(
            self.reference_point if self.reference_point is not None else _default_reference_point(F, maximise_flag),
            dtype=float,
        )
        samples = int(kwargs.get("samples", 10000))
        val = float(moocore.hv_approx(F, ref=ref, maximise=maximise_flag, nsamples=samples))
        return IndicatorResult(value=val, details={"reference_point": ref, "samples": samples})


@dataclass
class WHVRectIndicator(QualityIndicator):
    reference_point: np.ndarray | None = None
    rectangles: np.ndarray | None = None
    name: str = "whv_rect"

    def compute(
        self,
        front: np.ndarray,
        reference_front: np.ndarray | None = None,
        maximise: bool | Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> IndicatorResult:
        _require_moocore()
        moocore = _moocore  # local alias for typing
        assert moocore is not None  # guarded by _require_moocore()

        F = np.ascontiguousarray(front, dtype=float)
        maximise_flag = _normalize_maximise(maximise)
        ref = np.asarray(
            self.reference_point if self.reference_point is not None else _default_reference_point(F, maximise_flag),
            dtype=float,
        )
        rects_source = self.rectangles if self.rectangles is not None else kwargs.get("rectangles")
        if rects_source is None:
            raise ValueError("whv_rect requires rectangles to be provided.")
        rects = np.asarray(rects_source, dtype=float)
        if rects.size == 0:
            raise ValueError("whv_rect requires rectangles to be provided.")
        val = float(moocore.whv_rect(F, rects, ref=ref, maximise=maximise_flag))
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

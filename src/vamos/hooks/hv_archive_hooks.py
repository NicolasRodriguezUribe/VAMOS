from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

import numpy as np

from vamos.archive import BoundedArchive, BoundedArchiveConfig
from vamos.foundation.metrics.hypervolume import compute_hypervolume
from vamos.foundation.observer import RunContext
from vamos.monitoring import HVConvergenceConfig, HVConvergenceMonitor, HVDecision


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _try_compute_hv(F: np.ndarray, ref: Optional[List[float]] = None) -> Optional[float]:
    """
    Best-effort HV computation:
      - For 2D minimization: exact via compute_hypervolume
      - Otherwise: returns None
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    if F.ndim != 2 or F.shape[1] != 2:
        return None
    if ref is None:
        mx = np.max(F, axis=0)
        ref = [float(mx[0] + 1.0), float(mx[1] + 1.0)]
    try:
        return float(compute_hypervolume(F, ref))
    except Exception:
        return None


@dataclass
class HookManagerConfig:
    stopping_enabled: bool
    stop_cfg: HVConvergenceConfig
    archive_enabled: bool
    archive_cfg: BoundedArchiveConfig
    hv_ref_point: Optional[List[float]] = None


class CompositeLiveVisualization:
    """Fan-out live visualization events to multiple callbacks."""

    def __init__(self, callbacks: List[Any]) -> None:
        self._callbacks = [cb for cb in callbacks if cb is not None]

    def on_start(self, ctx: RunContext) -> None:
        for cb in self._callbacks:
            cb.on_start(ctx)

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        for cb in self._callbacks:
            cb.on_generation(generation, F=F, X=X, stats=stats)

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        for cb in self._callbacks:
            cb.on_end(final_F=final_F, final_stats=final_stats)

    def should_stop(self) -> bool:
        for cb in self._callbacks:
            should_stop = getattr(cb, "should_stop", None)
            if callable(should_stop):
                try:
                    if should_stop():
                        return True
                except Exception:
                    continue
        return False


class HookManager:
    """
    Orchestrates:
      - bounded archive updates
      - HV trace computation
      - HV convergence stopping
      - incremental artifact writing
    The runner calls `on_generation(...)` with eval counts in stats.
    """

    def __init__(self, out_dir: Path, cfg: HookManagerConfig):
        self.out_dir = out_dir
        self.cfg = cfg

        self.hv_trace_path = out_dir / "hv_trace.csv"
        self.archive_stats_path = out_dir / "archive_stats.csv"

        self.monitor = HVConvergenceMonitor(cfg.stop_cfg) if cfg.stopping_enabled else None
        self.archive = BoundedArchive(cfg.archive_cfg) if cfg.archive_enabled else None

        self._last_hv: Optional[float] = None
        self._last_sample_evals: Optional[int] = None
        self._stop_decision: Optional[HVDecision] = None

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if F is None:
            return
        evals = None
        if isinstance(stats, dict):
            evals = stats.get("evals")
        if evals is None:
            evals = generation
        self.on_checkpoint(evals=int(evals), F=np.asarray(F), X=None if X is None else np.asarray(X))

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        return None

    def _should_sample(self, evals: int) -> bool:
        if self.monitor is None:
            return True
        every_k = int(self.cfg.stop_cfg.every_k)
        if every_k <= 0:
            return True
        if self._last_sample_evals is None:
            return True
        return (evals - self._last_sample_evals) >= every_k

    def on_checkpoint(
        self,
        evals: int,
        F: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> None:
        # Update archive (optional)
        if self.archive is not None:
            upd = self.archive.add(X=X, F=F, evals=int(evals))
            _append_csv_row(
                self.archive_stats_path,
                fieldnames=["evals", "archive_size", "inserted", "pruned", "prune_reason"],
                row={
                    "evals": int(evals),
                    "archive_size": int(self.archive.size()),
                    "inserted": int(upd.inserted),
                    "pruned": int(upd.pruned),
                    "prune_reason": str(upd.prune_reason),
                },
            )

        if not self._should_sample(evals):
            return
        self._last_sample_evals = int(evals)

        # Choose set to compute HV on: archive if enabled else provided F
        hv_set = F
        if self.archive is not None and self.archive.size() > 0:
            hv_set = self.archive.F

        hv = _try_compute_hv(hv_set, ref=self.cfg.hv_ref_point)
        hv_delta = None
        stop_flag = False
        stop_reason = ""

        if hv is not None and self.monitor is not None:
            dec = self.monitor.add_sample(int(evals), float(hv))
            self._stop_decision = dec
            hv_delta = dec.hv_delta
            stop_flag = bool(dec.stop)
            stop_reason = str(dec.reason)
        elif hv is None and self.monitor is not None:
            stop_reason = "monitor_enabled_but_hv_unavailable"

        if hv_delta is None and hv is not None and self._last_hv is not None:
            hv_delta = float(hv - self._last_hv)

        reason = "hv_ok" if hv is not None else "hv_unavailable"
        if stop_reason:
            reason = stop_reason

        _append_csv_row(
            self.hv_trace_path,
            fieldnames=["evals", "hv", "hv_delta", "stop_flag", "reason"],
            row={
                "evals": int(evals),
                "hv": "" if hv is None else float(hv),
                "hv_delta": "" if hv_delta is None else float(hv_delta),
                "stop_flag": int(1 if stop_flag else 0),
                "reason": reason,
            },
        )

        if hv is not None:
            self._last_hv = float(hv)

    def should_stop(self) -> bool:
        if self._stop_decision is None:
            return False
        return bool(self._stop_decision.stop)

    def metadata_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"stopping": {}, "archive": {}}

        if self.monitor is None:
            payload["stopping"] = {"enabled": False}
        else:
            dec = self._stop_decision
            payload["stopping"] = {
                "enabled": True,
                "monitor_type": "hv_convergence",
                "params": self.cfg.stop_cfg.__dict__,
                "triggered": bool(dec.stop) if dec else False,
                "evals_stop": int(dec.evals) if (dec and dec.stop) else None,
                "reason": str(dec.reason) if dec else None,
            }

        if self.archive is None:
            payload["archive"] = {"enabled": False}
        else:
            payload["archive"] = {
                "enabled": True,
                "archive_type": self.cfg.archive_cfg.archive_type,
                "params": self.cfg.archive_cfg.__dict__,
                "final_size": int(self.archive.size()),
                "total_inserted": int(self.archive.total_inserted),
                "total_pruned": int(self.archive.total_pruned),
            }

        return payload

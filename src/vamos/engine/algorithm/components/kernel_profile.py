"""
Lightweight per-generation kernel profiling helpers.

This module is intentionally small: it uses perf_counter_ns() around hot calls
and stores aggregated timings so algorithm loops can report measured bottlenecks
without heavy instrumentation frameworks.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _parse_bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Expected one of {_TRUE_VALUES | _FALSE_VALUES}, got '{value}'.")


def _resolve_enabled(config: Mapping[str, Any] | None) -> bool:
    enabled: bool | None = None

    env_raw = os.environ.get("VAMOS_PROFILE_KERNELS")
    if env_raw is not None:
        enabled = _parse_bool_env(env_raw)

    cfg_raw: Any | None = None
    if config is not None:
        if "profile_kernels" in config:
            cfg_raw = config["profile_kernels"]
        elif "kernel_profile" in config:
            cfg_raw = config["kernel_profile"]

    if cfg_raw is not None:
        if isinstance(cfg_raw, Mapping):
            enabled = bool(cfg_raw.get("enabled", True))
        else:
            enabled = bool(cfg_raw)

    return bool(enabled)


@dataclass
class KernelProfiler:
    """
    Collect per-generation timings for hot algorithm phases.

    Timings are stored in nanoseconds and summarized in milliseconds.
    """

    rows: list[dict[str, int]] = field(default_factory=list)
    totals_ns: dict[str, int] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    _current_row: dict[str, int] | None = field(default=None, init=False, repr=False)

    def start_generation(self, *, generation: int | None = None, evaluations: int | None = None) -> None:
        row: dict[str, int] = {}
        if generation is not None:
            row["_generation"] = int(generation)
        if evaluations is not None:
            row["_evaluations"] = int(evaluations)
        self._current_row = row

    def end_generation(self, *, generation: int | None = None, evaluations: int | None = None) -> None:
        if self._current_row is None:
            row: dict[str, int] = {}
        else:
            row = self._current_row
        if generation is not None:
            row["_generation"] = int(generation)
        if evaluations is not None:
            row["_evaluations"] = int(evaluations)
        self.rows.append(row)
        self._current_row = None

    def add_ns(self, label: str, duration_ns: int) -> None:
        if duration_ns < 0:
            return
        ns = int(duration_ns)
        if self._current_row is None:
            self._current_row = {}
        self._current_row[label] = self._current_row.get(label, 0) + ns
        self.totals_ns[label] = self.totals_ns.get(label, 0) + ns
        self.call_counts[label] = self.call_counts.get(label, 0) + 1

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(label, time.perf_counter_ns() - start)

    def summary(self) -> dict[str, Any]:
        labels = sorted(k for k in self.totals_ns.keys() if not k.startswith("_"))
        n_generations = len(self.rows)
        per_kernel: dict[str, dict[str, float | int]] = {}

        for label in labels:
            samples_ns = [int(row.get(label, 0)) for row in self.rows]
            total_ns = int(sum(samples_ns))
            mean_ms = float(statistics.fmean(samples_ns) / 1e6) if samples_ns else 0.0
            std_ms = float(statistics.pstdev(samples_ns) / 1e6) if len(samples_ns) > 1 else 0.0
            per_kernel[label] = {
                "total_ms": total_ns / 1e6,
                "mean_ms_per_generation": mean_ms,
                "std_ms_per_generation": std_ms,
                "calls": int(self.call_counts.get(label, 0)),
            }

        total_wall_ns = int(sum(int(row.get("generation_total", 0)) for row in self.rows))
        return {
            "enabled": True,
            "n_generations": n_generations,
            "total_wall_ms": total_wall_ns / 1e6,
            "per_kernel": per_kernel,
        }


def resolve_kernel_profiler(config: Mapping[str, Any] | None) -> KernelProfiler | None:
    """
    Create a profiler when profiling is enabled via config or environment.

    - Env: VAMOS_PROFILE_KERNELS=1
    - Config: profile_kernels=True (or kernel_profile=True)
    """
    if not _resolve_enabled(config):
        return None
    return KernelProfiler()


__all__ = ["KernelProfiler", "resolve_kernel_profiler"]

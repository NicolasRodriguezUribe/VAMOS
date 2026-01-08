from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from vamos.foundation.observer import RunContext


class LiveVisualization(Protocol):
    """Callback interface for live/streaming visualization."""

    def on_start(self, ctx: RunContext) -> None: ...

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None: ...

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None: ...


class NoOpLiveVisualization:
    """Default no-op implementation."""

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        return None

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        return None


def _safe_matplotlib(interactive_backend: str | None = None):
    try:
        import matplotlib

        if interactive_backend:
            try:
                matplotlib.use(interactive_backend, force=True)
            except Exception:
                pass
        backend = matplotlib.get_backend().lower()
        if "agg" in backend and interactive_backend is None:
            pass
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


@dataclass
class LiveParetoPlot:
    """Interactive Pareto front plotter for 2D/3D objective spaces."""

    update_interval: int = 1
    max_points: int = 1000
    objectives_to_plot: Optional[Sequence[int]] = None
    interactive_backend: Optional[str] = None
    save_final_path: Optional[str | os.PathLike[str]] = None
    title: str = "Pareto front (live)"

    figure: Any = field(default=None, init=False)
    axes: Any = field(default=None, init=False)
    scatter: Any = field(default=None, init=False)
    last_update: int = field(default=-1, init=False)
    dims: int = field(default=2, init=False)
    plt: Any = field(default=None, init=False)
    _interactive: bool = field(default=False, init=False)

    def on_start(self, ctx: RunContext) -> None:
        self.plt = _safe_matplotlib(self.interactive_backend)
        if self.plt is None:
            return
        self.plt.ion()
        backend_name = str(self.plt.get_backend()).lower()
        self._interactive = not any(token in backend_name for token in ("agg", "inline", "pdf", "svg", "ps", "cairo"))
        problem = ctx.problem
        if problem is not None and getattr(problem, "n_obj", 2) >= 3:
            self.dims = 3
        self.figure = self.plt.figure(figsize=(6, 5))
        if self.dims == 3:
            self.axes = self.figure.add_subplot(111, projection="3d")
            self.axes.set_zlabel("Objective 3")
        else:
            self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel("Objective 1")
        self.axes.set_ylabel("Objective 2")
        self.axes.set_title(self.title)
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def _tick(self) -> None:
        if self.figure is None:
            return
        self.figure.canvas.draw_idle()
        flush = getattr(self.figure.canvas, "flush_events", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
        if self._interactive:
            self.plt.pause(0.01)

    def _maybe_subsample(self, F: np.ndarray) -> np.ndarray:
        if F.shape[0] <= self.max_points:
            return F
        idx = random.sample(range(F.shape[0]), self.max_points)
        return F[idx]

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.plt is None or F is None or generation - self.last_update < self.update_interval:
            return
        self.last_update = generation
        data = self._maybe_subsample(np.asarray(F, dtype=float))
        if data.ndim != 2 or data.shape[0] == 0:
            return
        dims = self.dims
        if self.objectives_to_plot is not None:
            idx = list(self.objectives_to_plot)[:dims]
        else:
            idx = list(range(min(dims, data.shape[1])))
        coords = data[:, idx]
        if self.scatter is None:
            if dims == 3:
                self.scatter = self.axes.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=18, alpha=0.8)
            else:
                (self.scatter,) = self.axes.plot(coords[:, 0], coords[:, 1], "o", markersize=4, alpha=0.8)
        else:
            if dims == 3:
                self.scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
            else:
                self.scatter.set_data(coords[:, 0], coords[:, 1])
        if stats and "hv" in stats:
            self.axes.set_title(f"{self.title} | HV={stats['hv']:.4f}")
        self._tick()

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.plt is None:
            return
        self.on_generation(self.last_update + 1, final_F, stats=final_stats)
        if self.save_final_path:
            path = os.fspath(self.save_final_path)
            dir_name = os.path.dirname(path) or "."
            os.makedirs(dir_name, exist_ok=True)
            self.figure.savefig(path, dpi=200)


@dataclass
class LiveTuningPlot:
    """Simple live scatter for tuning/meta-optimization results."""

    x_param: str
    y_metric: str = "hv"
    color_param: Optional[str] = None
    update_interval: int = 1
    save_final_path: Optional[str | os.PathLike[str]] = None
    interactive_backend: Optional[str] = None

    plt: Any = field(default=None, init=False)
    figure: Any = field(default=None, init=False)
    axes: Any = field(default=None, init=False)
    last_update: int = field(default=-1, init=False)
    _interactive: bool = field(default=False, init=False)

    def on_start(self, ctx: RunContext) -> None:
        self.plt = _safe_matplotlib(self.interactive_backend)
        if self.plt is None:
            return
        self.plt.ion()
        backend_name = str(self.plt.get_backend()).lower()
        self._interactive = not any(token in backend_name for token in ("agg", "inline", "pdf", "svg", "ps", "cairo"))
        self.figure, self.axes = self.plt.subplots(figsize=(6, 4))
        self.axes.set_xlabel(self.x_param)
        self.axes.set_ylabel(self.y_metric)
        self.figure.tight_layout()

    def _tick(self) -> None:
        if self.figure is None:
            return
        self.figure.canvas.draw_idle()
        flush = getattr(self.figure.canvas, "flush_events", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
        if self._interactive:
            self.plt.pause(0.01)

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.plt is None or stats is None:
            return
        if generation - self.last_update < self.update_interval:
            return
        self.last_update = generation
        x_vals = stats.get("x_vals")
        y_vals = stats.get("y_vals")
        colors = stats.get("colors")
        if x_vals is None or y_vals is None:
            return
        self.axes.clear()
        self.axes.set_xlabel(self.x_param)
        self.axes.set_ylabel(self.y_metric)
        self.axes.scatter(x_vals, y_vals, c=colors if colors is not None else "tab:blue", s=25, alpha=0.8)
        self._tick()

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.plt is None:
            return
        if self.save_final_path:
            path = os.fspath(self.save_final_path)
            dir_name = os.path.dirname(path) or "."
            os.makedirs(dir_name, exist_ok=True)
            self.figure.savefig(path, dpi=200)

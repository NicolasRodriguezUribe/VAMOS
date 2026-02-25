from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from vamos.foundation.observer import RunContext


class LiveVisualization(Protocol):
    """Callback interface for live/streaming visualization."""

    def on_start(self, ctx: RunContext | None = None) -> None: ...

    def on_generation(
        self,
        generation: int,
        F: np.ndarray | None = None,
        X: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None: ...

    def on_end(
        self,
        final_F: np.ndarray | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None: ...


class NoOpLiveVisualization:
    """Default no-op implementation."""

    def on_start(self, ctx: RunContext | None = None) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: np.ndarray | None = None,
        X: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_end(
        self,
        final_F: np.ndarray | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None


__all__ = ["LiveVisualization", "NoOpLiveVisualization"]

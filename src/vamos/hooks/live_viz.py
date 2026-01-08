from __future__ import annotations

from typing import Any, Optional, Protocol

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


__all__ = ["LiveVisualization", "NoOpLiveVisualization"]

"""
Metric tracking helpers (hypervolume, convergence).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from vamos.engine.algorithm.components.termination import HVTracker

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import KernelBackend


def setup_hv_tracker(
    hv_config: dict[str, Any] | None,
    kernel: KernelBackend | None,
) -> HVTracker:
    """
    Create HV tracker from config.

    Parameters
    ----------
    hv_config : dict[str, Any] | None
        HV termination configuration.
    kernel : KernelBackend | None
        Kernel backend.

    Returns
    -------
    HVTracker
        Configured tracker (may be disabled if config is None).
    """
    return HVTracker(hv_config, kernel)


__all__ = ["setup_hv_tracker"]

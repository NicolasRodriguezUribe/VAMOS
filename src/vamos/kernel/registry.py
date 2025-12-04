"""
Kernel backend registry.

This lightweight registry maps backend names to their kernel classes so that
orchestration layers can resolve kernels without hard-coding if/elif chains.
"""
from __future__ import annotations

from typing import Callable, Dict

from .moocore_backend import MooCoreKernel
from .numba_backend import NumbaKernel
from .numpy_backend import NumPyKernel

KERNELS: Dict[str, Callable[[], object]] = {
    "numpy": NumPyKernel,
    "numba": NumbaKernel,
    "moocore": MooCoreKernel,
}


def resolve_kernel(name: str):
    key = name.lower()
    try:
        return KERNELS[key]()
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown engine '{name}'. Available: {available}") from exc


__all__ = ["KERNELS", "resolve_kernel"]

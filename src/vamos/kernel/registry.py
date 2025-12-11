"""
Kernel backend registry.

This lightweight registry maps backend names to their kernel classes so that
orchestration layers can resolve kernels without hard-coding if/elif chains.
Backends that rely on optional dependencies (numba, moocore) are lazy-loaded so
`import vamos` remains safe on a minimal install.
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict

from .backend import KernelBackend
from .numpy_backend import NumPyKernel


def _load_numba():
    try:
        module = import_module("vamos.kernel.numba_backend")
        return module.NumbaKernel()
    except ImportError as exc:
        raise ImportError(
            "Kernel 'numba' requires the [backends] extra (numba>=0.57). "
            "Install with `pip install -e \".[backends]\"`."
        ) from exc


def _load_moocore():
    try:
        module = import_module("vamos.kernel.moocore_backend")
        return module.MooCoreKernel()
    except ImportError as exc:
        raise ImportError(
            "Kernel 'moocore' requires the [backends] extra (moocore>=0.4). "
            "Install with `pip install -e \".[backends]\"`."
        ) from exc


KERNELS: Dict[str, Callable[[], KernelBackend]] = {
    "numpy": NumPyKernel,
    "numba": _load_numba,
    "moocore": _load_moocore,
}


def resolve_kernel(name: str) -> KernelBackend:
    key = name.lower()
    try:
        factory = KERNELS[key]
    except KeyError as exc:
        available = ", ".join(sorted(KERNELS))
        raise ValueError(f"Unknown engine '{name}'. Available: {available}") from exc
    return factory()


__all__ = ["KERNELS", "resolve_kernel"]

"""
Foundation layer: backend kernels for vectorized compute-heavy operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import KernelBackend


def default_kernel() -> KernelBackend:
    """Return the best available kernel backend.

    Priority: CppBackend -> NumbaKernel -> NumPyKernel.
    """
    try:
        from .cpp_backend import CppBackend

        return CppBackend()
    except ImportError:
        pass

    try:
        from .numba_backend import NumbaKernel

        return NumbaKernel()
    except ImportError:
        from .numpy_backend import NumPyKernel

        return NumPyKernel()

"""
Public plotting namespace.

Re-exports plotting helpers from vamos.ux.plotting.
"""
from __future__ import annotations

from .ux import plotting as _plotting

__all__ = list(_plotting.__all__)

for _name in __all__:
    globals()[_name] = getattr(_plotting, _name)

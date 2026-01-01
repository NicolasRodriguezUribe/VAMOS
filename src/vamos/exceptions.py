"""
Public exceptions namespace.

Re-exports the canonical exception classes from vamos.foundation.exceptions.
"""

from __future__ import annotations

from .foundation import exceptions as _exceptions

__all__ = list(_exceptions.__all__)

for _name in __all__:
    globals()[_name] = getattr(_exceptions, _name)

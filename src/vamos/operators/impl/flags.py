from __future__ import annotations

import os

_NUMBA_VARIATION_OVERRIDE: bool | None = None


def set_numba_variation(enabled: bool | None) -> None:
    """Override Numba variation toggle (None defers to env)."""
    global _NUMBA_VARIATION_OVERRIDE
    if enabled is None:
        _NUMBA_VARIATION_OVERRIDE = None
    else:
        _NUMBA_VARIATION_OVERRIDE = bool(enabled)


def should_use_numba_variation() -> bool:
    """Return True if Numba variation ops should be used."""
    if _NUMBA_VARIATION_OVERRIDE is not None:
        return _NUMBA_VARIATION_OVERRIDE
    return os.environ.get("VAMOS_USE_NUMBA_VARIATION", "").lower() in {"1", "true", "yes"}


__all__ = ["set_numba_variation", "should_use_numba_variation"]

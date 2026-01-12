from __future__ import annotations

import logging


def configure_vamos_logging(*, level: int = logging.INFO) -> None:
    """
    Configure a minimal console logger for VAMOS.

    Notes:
        - This is intentionally opt-in (library code must not call logging.basicConfig()).
        - The handler is only attached if neither the root logger nor the "vamos" logger has handlers.
    """
    root = logging.getLogger()
    vamos_logger = logging.getLogger("vamos")

    # If the user already configured logging, don't interfere.
    if root.handlers or vamos_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    vamos_logger.addHandler(handler)
    vamos_logger.setLevel(level)
    vamos_logger.propagate = False


__all__ = ["configure_vamos_logging"]

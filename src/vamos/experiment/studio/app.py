"""
Entry point for the VAMOS Studio CLI.

Delegates to the UX-layer Streamlit app to keep UI code out of experiment modules.
"""

from __future__ import annotations

from vamos.ux.studio.app import main

__all__ = ["main"]

"""Compatibility CLI entry point for VAMOS Studio."""

from __future__ import annotations

from collections.abc import Sequence

from vamos.ux.panel.launcher import main as _panel_main


def main(argv: Sequence[str] | None = None) -> int:
    return _panel_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

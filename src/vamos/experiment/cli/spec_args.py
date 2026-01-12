from __future__ import annotations

import argparse
from typing import Any

_SPEC_KEY_ATTR = "_vamos_spec_key"


def add_spec_argument(parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> argparse.Action:
    """
    Register an argparse argument that can also be set via experiment spec defaults.

    The spec key is the argument's `dest` name.
    """
    action = parser.add_argument(*args, **kwargs)
    setattr(action, _SPEC_KEY_ATTR, action.dest)
    return action


def parser_spec_keys(parser: argparse.ArgumentParser) -> set[str]:
    """Return all spec-backed keys registered on this parser."""
    keys: set[str] = set()
    for action in parser._actions:  # argparse stores actions internally
        key = getattr(action, _SPEC_KEY_ATTR, None)
        if isinstance(key, str) and key:
            keys.add(key)
    return keys

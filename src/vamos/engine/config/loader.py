"""
Config loading utilities shared by CLI and programmatic entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_experiment_spec(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON experiment specification.
    """
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Config file '{spec_path}' does not exist.")
    suffix = spec_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("YAML config requested but PyYAML is not installed. Install with 'pip install pyyaml'.") from exc
        with spec_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    with spec_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

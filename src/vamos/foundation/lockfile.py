"""
Lockfile generation utilities for run reproducibility.
"""
from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import distributions, version, PackageNotFoundError
from pathlib import Path
from typing import Any


def generate_lockfile_data() -> dict[str, Any]:
    """
    Generate a dictionary containing environment information associated with a run.
    
    Returns:
        Dictionary with keys:
        - python: Python version info
        - platform: OS/System info
        - packages: Dictionary of installed packages and their versions
    """
    # Python and OS info
    env_info = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    }

    # Installed packages
    packages = {}
    for dist in distributions():
        try:
            packages[dist.metadata["Name"]] = dist.version
        except Exception:
            # Should not happen typically, but fail safe
            continue
            
    # Sort for deterministic output
    packages = dict(sorted(packages.items()))

    return {
        "environment": env_info,
        "packages": packages,
    }


def write_lockfile(path: str | Path) -> Path:
    """
    Generate and write environment lockfile to the specified path.
    
    Args:
        path: Output path for the lockfile.
        
    Returns:
        Path to the written file.
    """
    path = Path(path)
    # Ensure filename matches standard if directory provided?
    # Usually we expect access to full path or directory.
    # If directory, append 'vamos.lock'.
    if path.is_dir():
        path = path / "vamos.lock"
        
    data = generate_lockfile_data()
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return path


__all__ = ["generate_lockfile_data", "write_lockfile"]

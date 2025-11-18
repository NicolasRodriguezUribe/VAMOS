from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


_FILE_NAME_MAP = {
    "kroa100": "kroA100.tsp",
    "krob100": "kroB100.tsp",
    "kroc100": "kroC100.tsp",
    "krod100": "kroD100.tsp",
    "kroe100": "kroE100.tsp",
}


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    return project_root / "data" / "tsplib"


@lru_cache(maxsize=None)
def load_tsplib_coords(name: str) -> np.ndarray:
    """
    Parse a TSPLIB .tsp file and return coordinates as a NumPy array.
    Results are cached for repeated calls.
    """
    key = name.lower()
    filename = _FILE_NAME_MAP.get(key, f"{key}.tsp")
    path = _data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"TSPLIB instance '{name}' not found at {path}")

    coords: list[tuple[float, float]] = []
    in_section = False
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("NODE_COORD_SECTION"):
                in_section = True
                continue
            if upper.startswith("EOF"):
                break
            if not in_section:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                _, x_val, y_val = parts[:3]
                coords.append((float(x_val), float(y_val)))
            except ValueError as exc:  # pragma: no cover - corrupted data
                raise ValueError(f"Invalid coordinate line in {path}: {line}") from exc

    if not coords:
        raise ValueError(f"No coordinates found in TSPLIB file: {path}")

    return np.asarray(coords, dtype=float)

"""
Checkpointing utilities for saving and resuming optimization runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def save_checkpoint(
    path: str | Path,
    *,
    X: np.ndarray,
    F: np.ndarray,
    generation: int,
    n_eval: int,
    rng_state: dict[str, Any],
    G: np.ndarray | None = None,
    archive_X: np.ndarray | None = None,
    archive_F: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save algorithm state to a checkpoint file.

    Args:
        path: File path for checkpoint (will add .ckpt extension if missing).
        X: Population decision variables.
        F: Population objective values.
        generation: Current generation number.
        n_eval: Total evaluations so far.
        rng_state: RNG state from `rng.bit_generator.state`.
        G: Constraint values (optional).
        archive_X: Archive decision variables (optional).
        archive_F: Archive objective values (optional).
        extra: Additional algorithm-specific state (optional).

    Returns:
        Path to saved checkpoint file.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".ckpt")

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": 1,
        "X": X,
        "F": F,
        "G": G,
        "generation": generation,
        "n_eval": n_eval,
        "rng_state": rng_state,
        "archive_X": archive_X,
        "archive_F": archive_F,
        "extra": extra or {},
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """
    Load algorithm state from a checkpoint file.

    Args:
        path: Path to checkpoint file.

    Returns:
        Dictionary containing checkpoint data with keys:
        - X, F, G: Population arrays
        - generation, n_eval: Progress counters
        - rng_state: RNG state dict
        - archive_X, archive_F: Archive arrays (may be None)
        - extra: Additional state dict

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        ValueError: If checkpoint version is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        checkpoint = cast(dict[str, Any], pickle.load(f))

    version = checkpoint.get("version", 0)
    if version != 1:
        raise ValueError(f"Unsupported checkpoint version: {version}")

    return checkpoint


def restore_rng(rng: "Generator", state: dict[str, Any]) -> None:
    """
    Restore RNG state from checkpoint.

    Args:
        rng: NumPy random generator to restore.
        state: State dict from checkpoint['rng_state'].
    """
    rng.bit_generator.state = state


__all__ = ["save_checkpoint", "load_checkpoint", "restore_rng"]

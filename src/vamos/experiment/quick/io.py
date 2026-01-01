from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from vamos.foundation.problem.types import ProblemProtocol

logger = logging.getLogger(__name__)


def save_quick_result(
    path: str,
    *,
    F: np.ndarray,
    X: np.ndarray | None,
    problem: ProblemProtocol,
    algorithm: str,
    n_evaluations: int,
    seed: int,
) -> None:
    """Save quick results to a directory with CSV data and JSON metadata."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "FUN.csv", F, delimiter=",")
    if X is not None:
        np.savetxt(out_dir / "X.csv", X, delimiter=",")

    metadata: dict[str, Any] = {
        "algorithm": algorithm,
        "n_evaluations": n_evaluations,
        "seed": seed,
        "n_solutions": int(F.shape[0]),
        "n_objectives": int(F.shape[1]) if F.ndim > 1 else 1,
        "problem": type(problem).__name__,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Results saved to %s", out_dir)

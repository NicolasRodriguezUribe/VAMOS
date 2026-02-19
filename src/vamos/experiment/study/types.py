from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.archive import ExternalArchiveConfig
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection


@dataclass(frozen=True)
class StudyTask:
    """
    Defines a single algorithm/engine/problem/seed combination.
    """

    algorithm: str
    engine: str
    problem: str
    n_var: int | None = None
    n_obj: int | None = None
    seed: int = ExperimentConfig().seed
    selection_pressure: int = 2
    external_archive: ExternalArchiveConfig | None = None
    nsgaii_variation: dict[str, Any] | None = None
    moead_variation: dict[str, Any] | None = None
    smsemoa_variation: dict[str, Any] | None = None
    config_overrides: dict[str, Any] | None = None


@dataclass
class StudyResult:
    task: StudyTask
    selection: ProblemSelection
    metrics: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        hv_ref = self.metrics.get("hv_reference")
        hv_ref_str = " ".join(f"{val:.6f}" for val in hv_ref) if isinstance(hv_ref, np.ndarray) else ""
        row = {
            "problem": self.selection.spec.key,
            "problem_label": self.selection.spec.label,
            "n_var": self.selection.n_var,
            "n_obj": self.selection.n_obj,
            "algorithm": self.metrics["algorithm"],
            "engine": self.metrics["engine"],
            "seed": self.task.seed,
            "time_ms": self.metrics["time_ms"],
            "evaluations": self.metrics["evaluations"],
            "evals_per_sec": self.metrics["evals_per_sec"],
            "spread": self.metrics.get("spread"),
            "hv": self.metrics.get("hv"),
            "hv_source": self.metrics.get("hv_source"),
            "hv_reference": hv_ref_str,
            "backend_device": self.metrics.get("backend_device"),
            "backend_capabilities": ",".join(self.metrics.get("backend_capabilities", [])),
            "output_dir": self.metrics.get("output_dir"),
        }
        for name, value in (self.metrics.get("indicator_values") or {}).items():
            row[f"indicator_{name}"] = value
        return row

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.engine.archive import ExternalArchiveConfig
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
            "algorithm_base": self.metrics.get("algorithm_base", self.task.algorithm),
            "engine": self.metrics["engine"],
            "seed": self.task.seed,
            "time_ms": self.metrics["time_ms"],
            "evaluations": self.metrics["evaluations"],
            "evals_per_sec": self.metrics["evals_per_sec"],
            "spread": self.metrics.get("spread"),
            "hv": self.metrics.get("hv"),
            "hv_source": self.metrics.get("hv_source"),
            "archive_subset_hv": self.metrics.get("archive_subset_hv"),
            "archive_subset_hv_source": self.metrics.get("archive_subset_hv_source"),
            "hv_reference": hv_ref_str,
            "backend_device": self.metrics.get("backend_device"),
            "backend_capabilities": ",".join(self.metrics.get("backend_capabilities", [])),
            "output_dir": self.metrics.get("output_dir"),
            "archive_mode": self.metrics.get("archive_mode"),
            "archive_execution_mode": self.metrics.get("archive_execution_mode"),
            "archive_survival_path": self.metrics.get("archive_survival_path"),
            "archive_present": self.metrics.get("archive_present"),
            "archive_size": self.metrics.get("archive_size"),
            "archive_subset_size": self.metrics.get("archive_subset_size"),
            "archive_subset_selector": self.metrics.get("archive_subset_selector"),
            "hybrid_status": self.metrics.get("hybrid_status"),
            "hybrid_fallback_reason": self.metrics.get("hybrid_fallback_reason"),
            "hybrid_split_front_mode": self.metrics.get("hybrid_split_front_mode"),
            "hybrid_split_front_reason": self.metrics.get("hybrid_split_front_reason"),
            "hybrid_generations": self.metrics.get("hybrid_generations"),
            "hybrid_archive_reference_generations": self.metrics.get("hybrid_archive_reference_generations"),
            "hybrid_local_only_generations": self.metrics.get("hybrid_local_only_generations"),
            "hybrid_no_split_generations": self.metrics.get("hybrid_no_split_generations"),
        }
        for name, value in (self.metrics.get("indicator_values") or {}).items():
            row[f"indicator_{name}"] = value
        for name, value in (self.metrics.get("archive_subset_indicator_values") or {}).items():
            row[f"archive_subset_{name}"] = value
        return row

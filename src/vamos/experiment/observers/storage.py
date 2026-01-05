from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vamos.foundation.observer import Observer, RunContext
from vamos.foundation.core.io_utils import ensure_dir, write_population, write_metadata, write_timing
from vamos.foundation.core.metadata import build_run_metadata
from vamos.adaptation.aos.logging import write_aos_summary, write_aos_trace


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _project_root() -> Path:
    # Adjust as needed (assuming src/vamos/experiment/observers/storage.py)
    # Root is typically 4 levels up from this file
    return Path(__file__).resolve().parents[4]


class StorageObserver(Observer):
    """
    Observer responsible for persisting run artifacts to disk.
    Migrated from vamos.experiment.runner_output.persist_run_outputs.
    """

    def __init__(
        self,
        output_dir: str,
        # Capture all the misc kwargs that were passed to persist_run_outputs
        # Ideally, many of these should be in RunContext or final_stats
        project_root: Path | None = None,
        config_source: str | None = None,
        problem_override: dict | None = None,
        hv_stop_config: dict | None = None,
        selection_pressure: int = 2,
        external_archive_size: int | None = None,
        variations: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.project_root = project_root or _project_root()
        self.config_source = config_source
        self.problem_override = problem_override
        self.hv_stop_config = hv_stop_config
        self.selection_pressure = selection_pressure
        self.external_archive_size = external_archive_size
        self.variations = variations or {}

    def on_start(self, ctx: RunContext) -> None:
        ensure_dir(self.output_dir)

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        pass

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        if final_stats is None:
            return

        # Extract payload from statistics (assuming orchestrator puts things there)
        # In the new design, 'final_stats' should ideally contain what 'metrics' + 'payload' used to have.
        
        # NOTE: logic requires `payload` dict which contains 'archive', 'genealogy', etc.
        # We assume the caller (orchestrator) merges payload into final_stats or we access it differently.
        # For now, let's assume `final_stats` IS the rich metrics dictionary.
        
        payload = final_stats.get("payload", {})
        # If payload is missing, we might only have metrics.
        # For full persistence, we need X, G, Archive, etc.
        
        # Reconstruct F/X if passed directly
        F_to_save = final_F if final_F is not None else payload.get("F")
        X_to_save = payload.get("X")
        G_to_save = payload.get("G")
        archive = payload.get("archive")
        
        artifacts = write_population(
            self.output_dir,
            F_to_save,
            archive,
            X=X_to_save,
            G=G_to_save,
        )

        if payload.get("genealogy"):
            genealogy_path = self.output_dir / "genealogy.json"
            genealogy_path.write_text(json.dumps(payload["genealogy"], indent=2), encoding="utf-8")
            artifacts["genealogy"] = genealogy_path.name

        autodiff_info = final_stats.get("autodiff_info")
        if autodiff_info is not None:
            autodiff_path = self.output_dir / "autodiff_constraints.json"
            autodiff_path.write_text(json.dumps(autodiff_info, indent=2), encoding="utf-8")
            artifacts["autodiff_constraints"] = autodiff_path.name
            
        aos_payload = payload.get("aos") or {}
        trace_rows = aos_payload.get("trace_rows")
        summary_rows = aos_payload.get("summary")
        if trace_rows and summary_rows:
            trace_path = self.output_dir / "aos_trace.csv"
            summary_path = self.output_dir / "aos_summary.csv"
            write_aos_trace(trace_path, trace_rows)
            write_aos_summary(summary_path, summary_rows)
            artifacts["aos_trace"] = trace_path.name
            artifacts["aos_summary"] = summary_path.name

        total_time_ms = final_stats.get("time_ms", 0.0)
        write_timing(self.output_dir, total_time_ms)

        # Context needed for metadata
        # We need access to ctx from on_start? Or pass it here?
        # Observer protocol on_end doesn't take ctx.
        # We should store ctx in on_start.
        if not hasattr(self, "_ctx"):
            _logger().warning("StorageObserver.on_end called without on_start (missing context). Metadata might be incomplete.")
            return

        ctx = self._ctx
        
        # Build metadata
        metadata = build_run_metadata(
            ctx.selection,
            ctx.algorithm_name,
            ctx.engine_name,
            final_stats.get("config"), # cfg_data
            final_stats, # metrics
            kernel_backend=final_stats.get("_kernel_backend"),
            seed=getattr(ctx.config, "seed", None),
            config=ctx.config,
            project_root=self.project_root,
        )
        
        metadata["config_source"] = self.config_source
        if self.problem_override:
            metadata["problem_override"] = self.problem_override
        if self.hv_stop_config:
            metadata["hv_stop_config"] = self.hv_stop_config
            
        # Hook payload integration?
        # If HookManager was used, it should have injected its data into payload or metrics.
        # In the new design, HookManager is an Observer too!
        # It should expose its data differently.
        # For now, if payload has hook data, use it.
        
        if payload.get("genealogy"):
             metadata["genealogy"] = payload["genealogy"]
        if autodiff_info is not None:
             metadata["autodiff_constraints"] = autodiff_info

        # Artifact entries
        artifact_entries = {
            "fun": artifacts.get("fun"),
            "x": artifacts.get("x"),
            "g": artifacts.get("g"),
            "archive_fun": artifacts.get("archive_fun"),
            "archive_x": artifacts.get("archive_x"),
            "archive_g": artifacts.get("archive_g"),
            "genealogy": artifacts.get("genealogy"),
            "autodiff_constraints": artifacts.get("autodiff_constraints"),
            "aos_trace": artifacts.get("aos_trace"),
            "aos_summary": artifacts.get("aos_summary"),
            "time_ms": "time.txt",
        }
        hv_trace = self.output_dir / "hv_trace.csv"
        if hv_trace.exists():
            artifact_entries["hv_trace"] = hv_trace.name
        archive_stats = self.output_dir / "archive_stats.csv"
        if archive_stats.exists():
            artifact_entries["archive_stats"] = archive_stats.name
        metadata["artifacts"] = {k: v for k, v in artifact_entries.items() if v is not None}
        
        # Resolved Config
        problem_key = getattr(getattr(ctx.selection, "spec", None), "key", "unknown")
        encoding = getattr(ctx.problem, "encoding", "continuous")
        
        resolved_cfg = {
            "algorithm": ctx.algorithm_name,
            "engine": ctx.engine_name,
            "problem": problem_key,
            "n_var": getattr(ctx.selection, "n_var", getattr(ctx.problem, "n_var", None)),
            "n_obj": getattr(ctx.selection, "n_obj", getattr(ctx.problem, "n_obj", None)),
            "encoding": encoding,
            "population_size": getattr(ctx.config, "population_size", None),
            "offspring_population_size": getattr(ctx.config, "offspring_size", lambda: None)(),
            "max_evaluations": getattr(ctx.config, "max_evaluations", None),
            "seed": getattr(ctx.config, "seed", None),
            "selection_pressure": self.selection_pressure,
            "external_archive_size": self.external_archive_size,
            "hv_threshold": self.hv_stop_config.get("threshold_fraction") if self.hv_stop_config else None,
            "hv_reference_point": self.hv_stop_config.get("reference_point") if self.hv_stop_config else None,
            "hv_reference_front": self.hv_stop_config.get("reference_front_path") if self.hv_stop_config else None,
            # Variations map
            **self.variations,
            "config_source": self.config_source,
            "problem_override": self.problem_override,
        }
        
        # AOS config if present
        aos_config = getattr(ctx, "aos_config", None)
        if aos_config is not None:
             resolved_cfg["adaptive_operator_selection"] = aos_config
             
        write_metadata(self.output_dir, metadata, resolved_cfg)

        # Generate lockfile (Phase 4.3)
        try:
            from vamos.foundation.lockfile import write_lockfile
            write_lockfile(self.output_dir / "vamos.lock")
        except Exception as exc:
            _logger().warning("Failed to emit lockfile: %s", exc)

        _logger().info("Results stored in: %s", self.output_dir)

    def on_start(self, ctx: RunContext) -> None:
        self._ctx = ctx
        ensure_dir(self.output_dir)

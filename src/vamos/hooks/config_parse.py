from __future__ import annotations

from typing import Any

from vamos.archive import BoundedArchiveConfig
from vamos.monitoring import HVConvergenceConfig


def build_hv_stop_cfg(d: dict[str, Any]) -> HVConvergenceConfig:
    base = HVConvergenceConfig()
    data = dict(base.__dict__)
    data.update({k: v for k, v in (d or {}).items() if k in data})
    return HVConvergenceConfig(**data)


def build_archive_cfg(d: dict[str, Any]) -> BoundedArchiveConfig:
    base = BoundedArchiveConfig()
    data = dict(base.__dict__)
    data.update({k: v for k, v in (d or {}).items() if k in data})
    return BoundedArchiveConfig(**data)


def _extract_block(spec: dict[str, Any], key: str, problem_key: str | None) -> dict[str, Any]:
    block: dict[str, Any] = {}
    if isinstance(spec.get(key), dict):
        block = dict(spec[key])
    defaults = spec.get("defaults")
    if not block and isinstance(defaults, dict) and isinstance(defaults.get(key), dict):
        block = dict(defaults[key])
    if problem_key:
        problems = spec.get("problems")
        if isinstance(problems, dict):
            p_cfg = problems.get(problem_key)
            if isinstance(p_cfg, dict) and isinstance(p_cfg.get(key), dict):
                override = dict(p_cfg[key])
                merged = dict(block)
                merged.update(override)
                block = merged
    return block


def parse_stopping_archive(spec: dict[str, Any], problem_key: str | None = None) -> dict[str, Any]:
    """
    Reads an experiment spec dict and returns:
      stopping_enabled, stop_cfg, archive_enabled, archive_cfg, hv_ref_point
    """
    if not isinstance(spec, dict):
        spec = {}

    stopping = _extract_block(spec, "stopping", problem_key)
    archive = _extract_block(spec, "archive", problem_key)

    hv_block = stopping.get("hv_convergence", {}) if isinstance(stopping, dict) else {}
    stop_enabled = bool(hv_block.get("enabled", False))
    hv_ref_point = hv_block.get("ref_point")
    if isinstance(hv_ref_point, str) and hv_ref_point.lower() == "auto":
        hv_ref_point = None
    stop_cfg = build_hv_stop_cfg({k: v for k, v in hv_block.items() if k not in ("enabled", "ref_point")})

    arch_block = archive.get("bounded", {}) if isinstance(archive, dict) else {}
    arch_enabled = bool(arch_block.get("enabled", False))
    arch_cfg = build_archive_cfg({k: v for k, v in arch_block.items() if k != "enabled"})

    return {
        "stopping_enabled": stop_enabled,
        "stop_cfg": stop_cfg,
        "archive_enabled": arch_enabled,
        "archive_cfg": arch_cfg,
        "hv_ref_point": hv_ref_point,
    }

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from vamos.archive import BoundedArchiveConfig
from vamos.monitoring import HVConvergenceConfig


class StoppingArchiveConfig(TypedDict):
    stopping_enabled: bool
    stop_cfg: HVConvergenceConfig
    archive_enabled: bool
    archive_cfg: BoundedArchiveConfig
    hv_ref_point: list[float] | None


def build_hv_stop_cfg(d: Mapping[str, object] | None) -> HVConvergenceConfig:
    base = HVConvergenceConfig()
    data = dict(base.__dict__)
    data.update({k: v for k, v in (d or {}).items() if k in data})
    return HVConvergenceConfig(**data)


def build_archive_cfg(d: Mapping[str, object] | None) -> BoundedArchiveConfig:
    base = BoundedArchiveConfig()
    data = dict(base.__dict__)
    data.update({k: v for k, v in (d or {}).items() if k in data})
    return BoundedArchiveConfig(**data)


def _extract_block(spec: Mapping[str, object], key: str, problem_key: str | None) -> dict[str, object]:
    block: dict[str, object] = {}
    value = spec.get(key)
    if isinstance(value, Mapping):
        block = dict(value)
    defaults = spec.get("defaults")
    if not block and isinstance(defaults, Mapping):
        defaults_block = defaults.get(key)
        if isinstance(defaults_block, Mapping):
            block = dict(defaults_block)
    if problem_key:
        problems = spec.get("problems")
        if isinstance(problems, Mapping):
            p_cfg = problems.get(problem_key)
            if isinstance(p_cfg, Mapping):
                override_block = p_cfg.get(key)
                if isinstance(override_block, Mapping):
                    override = dict(override_block)
                    merged = dict(block)
                    merged.update(override)
                    block = merged
    return block


def parse_stopping_archive(spec: Mapping[str, object] | None, problem_key: str | None = None) -> StoppingArchiveConfig:
    """
    Reads an experiment spec dict and returns:
      stopping_enabled, stop_cfg, archive_enabled, archive_cfg, hv_ref_point
    """
    if not isinstance(spec, Mapping):
        spec = {}

    stopping = _extract_block(spec, "stopping", problem_key)
    archive = _extract_block(spec, "archive", problem_key)

    hv_raw = stopping.get("hv_convergence") if isinstance(stopping, Mapping) else None
    hv_block = hv_raw if isinstance(hv_raw, Mapping) else {}
    stop_enabled = bool(hv_block.get("enabled", False))
    hv_ref_point = hv_block.get("ref_point")
    if isinstance(hv_ref_point, str) and hv_ref_point.lower() == "auto":
        hv_ref_point = None
    stop_cfg = build_hv_stop_cfg({k: v for k, v in hv_block.items() if k not in ("enabled", "ref_point")})

    arch_raw = archive.get("bounded") if isinstance(archive, Mapping) else None
    arch_block = arch_raw if isinstance(arch_raw, Mapping) else {}
    arch_enabled = bool(arch_block.get("enabled", False))
    arch_cfg = build_archive_cfg({k: v for k, v in arch_block.items() if k != "enabled"})

    return {
        "stopping_enabled": stop_enabled,
        "stop_cfg": stop_cfg,
        "archive_enabled": arch_enabled,
        "archive_cfg": arch_cfg,
        "hv_ref_point": hv_ref_point,
    }


__all__ = ["StoppingArchiveConfig", "parse_stopping_archive"]

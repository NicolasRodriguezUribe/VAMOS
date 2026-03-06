from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

from vamos.engine.archive import BoundedArchiveConfig, PrunePolicy
from vamos.engine.hooks.hv_convergence import HVConvergenceConfig


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
    if d is None:
        return BoundedArchiveConfig()
    data = dict(d)
    if "archive_type" in data:
        raise TypeError("build_archive_cfg() does not accept legacy 'archive_type'; use 'size_cap' and 'prune_policy'.")

    size_cap = data.get("size_cap", 200)
    if not isinstance(size_cap, int):
        raise TypeError("archive.size_cap must be an integer.")

    truncate_size = data.get("truncate_size")
    if truncate_size is not None and not isinstance(truncate_size, int):
        raise TypeError("archive.truncate_size must be an integer or null.")

    epsilon = data.get("epsilon", 0.01)
    if not isinstance(epsilon, (int, float)):
        raise TypeError("archive.epsilon must be numeric.")

    hv_samples = data.get("hv_samples", 20000)
    if not isinstance(hv_samples, int):
        raise TypeError("archive.hv_samples must be an integer.")

    rng_seed = data.get("rng_seed", 0)
    if not isinstance(rng_seed, int):
        raise TypeError("archive.rng_seed must be an integer.")

    hv_ref_point_raw = data.get("hv_ref_point")
    hv_ref_point: list[float] | None
    if hv_ref_point_raw is None:
        hv_ref_point = None
    elif isinstance(hv_ref_point_raw, list):
        hv_ref_point = [float(value) for value in hv_ref_point_raw]
    else:
        raise TypeError("archive.hv_ref_point must be a list of floats or null.")

    prune_policy = data.get("prune_policy", "crowding")

    if not isinstance(prune_policy, str):
        raise TypeError("archive.prune_policy must be a string.")

    return BoundedArchiveConfig(
        enabled=bool(data.get("enabled", True)),
        nondominated_only=bool(data.get("nondominated_only", True)),
        size_cap=size_cap,
        truncate_size=truncate_size,
        epsilon=float(epsilon),
        prune_policy=cast(PrunePolicy, prune_policy),
        hv_ref_point=hv_ref_point,
        hv_samples=hv_samples,
        rng_seed=rng_seed,
    )


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

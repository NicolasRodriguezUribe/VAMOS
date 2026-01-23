"""
Algorithm-specific builder helpers to keep the factory slim.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.data import weight_path
from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.algorithm.config import (
    AlgorithmConfigProtocol,
    NSGAIIConfig,
    MOEADConfig,
    SMSEMOAConfig,
    NSGAIIIConfig,
    SPEA2Config,
    IBEAConfig,
    SMPSOConfig,
    AGEMOEAConfig,
    RVEAConfig,
)
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.algorithm.components.variation.helpers import ensure_supported_repair_name
from vamos.engine.config.variation import ensure_operator_tuple, merge_variation_overrides, resolve_default_variation_config


def _as_int(value: object) -> int:
    return int(cast(int | float | str, value))


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


def build_nsgaii_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    offspring_size: int,
    selection_pressure: int,
    external_archive_size: int | None,
    archive_type: str,
    nsgaii_variation: dict[str, Any] | None,
    track_genealogy: bool,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    var_cfg = resolve_default_variation_config(encoding, nsgaii_variation)

    builder = NSGAIIConfig.builder()
    builder.pop_size(pop_size)
    builder.offspring_size(offspring_size)
    builder.result_mode("population")

    if "crossover" in var_cfg:
        c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
        builder.crossover(c_name, **c_kwargs)
    if "mutation" in var_cfg:
        m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
        builder.mutation(m_name, **m_kwargs)
    if "selection" in var_cfg:
        s_name, s_kwargs = ensure_operator_tuple(var_cfg["selection"], key="selection")
        builder.selection(s_name, **s_kwargs)
    else:
        builder.selection("tournament", pressure=selection_pressure)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    if "adaptive_operator_selection" in var_cfg:
        aos_cfg = var_cfg["adaptive_operator_selection"]
        if not isinstance(aos_cfg, Mapping):
            raise ValueError("adaptive_operator_selection must be a mapping.")
        builder.adaptive_operator_selection(dict(aos_cfg))

    steady_state = bool(var_cfg.get("steady_state", False))
    replacement_size = var_cfg.get("replacement_size")
    if replacement_size is not None and not steady_state:
        steady_state = True
    if steady_state:
        builder.steady_state(True)
    if replacement_size is not None:
        builder.replacement_size(_as_int(replacement_size))

    if external_archive_size:
        builder.archive(external_archive_size)
        builder.archive_type(archive_type)
        builder.result_mode("external_archive")
    if track_genealogy:
        builder.track_genealogy(True)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("nsgaii")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_moead_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    moead_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    moead_overrides = moead_variation or {}
    var_cfg = resolve_default_variation_config(encoding, moead_overrides)
    if encoding == "real":
        if "crossover" not in moead_overrides:
            var_cfg["crossover"] = ("de", {"cr": 1.0, "f": 0.5})
        if "mutation" not in moead_overrides:
            var_cfg["mutation"] = ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})
    # Ensure default aggregation if not present
    if "aggregation" not in var_cfg:
        var_cfg["aggregation"] = ("pbi", {"theta": 5.0})

    # Merge any other overrides that resolve_default didn't catch (like aggregation)
    extra_cfg = {k: v for k, v in moead_overrides.items() if k not in var_cfg}
    var_cfg.update(extra_cfg)

    builder = MOEADConfig.builder()
    builder.pop_size(pop_size)
    builder.neighbor_size(20)
    builder.delta(0.9)
    builder.replace_limit(2)
    if "weight_vectors" in moead_overrides and moead_overrides["weight_vectors"] is not None:
        weight_cfg = moead_overrides["weight_vectors"]
        if isinstance(weight_cfg, dict):
            builder.weight_vectors(**weight_cfg)
        else:
            builder.weight_vectors(path=str(weight_cfg))
    else:
        builder.weight_vectors(path=str(weight_path("W3D_91.dat").parent))

    c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
    builder.mutation(m_name, **m_kwargs)

    if "aggregation" in var_cfg:
        a_name, a_kwargs = ensure_operator_tuple(var_cfg["aggregation"], key="aggregation")
        builder.aggregation(a_name, **a_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("moead")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_smsemoa_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    smsemoa_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    smsemoa_overrides = smsemoa_variation or {}
    var_cfg = resolve_default_variation_config(encoding, smsemoa_overrides)
    extra_cfg = {k: v for k, v in smsemoa_overrides.items() if k not in var_cfg}
    var_cfg.update(extra_cfg)

    builder = SMSEMOAConfig.builder()
    builder.pop_size(pop_size)

    c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
    builder.mutation(m_name, **m_kwargs)

    # SMS-EMOA typically uses random selection for steady-state
    s_name, s_kwargs = ensure_operator_tuple(var_cfg.get("selection", ("random", {})), key="selection")
    builder.selection(s_name, **s_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("smsemoa")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_nsgaiii_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    nsgaiii_variation: dict[str, Any] | None,
    selection_pressure: int,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    nsgaiii_overrides = nsgaiii_variation or {}
    var_cfg = resolve_default_variation_config(encoding, nsgaiii_overrides)
    extra_cfg = {k: v for k, v in nsgaiii_overrides.items() if k not in var_cfg}
    var_cfg.update(extra_cfg)

    builder = NSGAIIIConfig.builder()
    builder.pop_size(pop_size)

    c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
    builder.mutation(m_name, **m_kwargs)

    s_name, s_kwargs = ensure_operator_tuple(
        var_cfg.get("selection", ("tournament", {"pressure": selection_pressure})),
        key="selection",
    )
    builder.selection(s_name, **s_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("nsgaiii")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_spea2_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    selection_pressure: int,
    external_archive_size: int | None,
    spea2_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    spea2_overrides = spea2_variation or {}
    var_cfg = resolve_default_variation_config(encoding, spea2_overrides)
    extra_cfg = {k: v for k, v in spea2_overrides.items() if k not in {"crossover", "mutation", "repair"}}
    var_cfg.update(extra_cfg)

    builder = SPEA2Config.builder()
    builder.pop_size(pop_size)
    archive_override = var_cfg.get("archive_size")
    builder.archive_size(_as_int(archive_override) if archive_override is not None else (external_archive_size or pop_size))
    if "k_neighbors" in var_cfg and var_cfg["k_neighbors"] is not None:
        builder.k_neighbors(_as_int(var_cfg["k_neighbors"]))

    c_name, c_kwargs = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {"prob": 1.0, "eta": 20.0})), key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(
        var_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    builder.mutation(m_name, **m_kwargs)

    sel_name, sel_kwargs = ensure_operator_tuple(
        var_cfg.get("selection", ("tournament", {"pressure": selection_pressure})),
        key="selection",
    )
    builder.selection(sel_name, **sel_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("spea2")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_ibea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    selection_pressure: int,
    ibea_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    ibea_overrides = ibea_variation or {}
    var_cfg = resolve_default_variation_config(encoding, ibea_overrides)
    extra_cfg = {k: v for k, v in ibea_overrides.items() if k not in {"crossover", "mutation", "repair"}}
    var_cfg.update(extra_cfg)

    builder = IBEAConfig.builder()
    builder.pop_size(pop_size)
    c_name, c_kwargs = ensure_operator_tuple(var_cfg.get("crossover", ("sbx", {"prob": 1.0, "eta": 20.0})), key="crossover")
    builder.crossover(c_name, **c_kwargs)
    m_name, m_kwargs = ensure_operator_tuple(
        var_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    builder.mutation(m_name, **m_kwargs)
    sel_name, sel_kwargs = ensure_operator_tuple(
        var_cfg.get("selection", ("tournament", {"pressure": selection_pressure})),
        key="selection",
    )
    builder.selection(sel_name, **sel_kwargs)
    indicator = var_cfg.get("indicator")
    if indicator is None:
        builder.indicator("eps")
    elif isinstance(indicator, str):
        builder.indicator(indicator)
    else:
        raise ValueError("IBEA indicator must be a string.")
    builder.kappa(_as_float(var_cfg.get("kappa", 1.0)))
    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)
    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("ibea")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_smpso_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    external_archive_size: int | None,
    smpso_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    mut_cfg = merge_variation_overrides(
        {
            "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
        },
        smpso_variation,
    )
    builder = SMPSOConfig.builder()
    builder.pop_size(pop_size)
    builder.archive_size(pop_size if external_archive_size is None else external_archive_size)
    m_name, m_kwargs = ensure_operator_tuple(
        mut_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})),
        key="mutation",
    )
    builder.mutation(m_name, **m_kwargs)
    if "inertia" in mut_cfg:
        builder.inertia(_as_float(mut_cfg["inertia"]))
    if "c1" in mut_cfg:
        builder.c1(_as_float(mut_cfg["c1"]))
    if "c2" in mut_cfg:
        builder.c2(_as_float(mut_cfg["c2"]))
    if "vmax_fraction" in mut_cfg:
        builder.vmax_fraction(_as_float(mut_cfg["vmax_fraction"]))
    if "repair" in mut_cfg:
        r_name, r_kwargs = ensure_operator_tuple(mut_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)
    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("smpso")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


class DictConfigWrapper:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return self._data


def build_agemoea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    agemoea_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    agemoea_overrides = agemoea_variation or {}
    var_cfg = resolve_default_variation_config(encoding, agemoea_overrides)
    extra_cfg = {k: v for k, v in agemoea_overrides.items() if k not in var_cfg}
    var_cfg.update(extra_cfg)

    if encoding == "real":
        if "crossover" not in agemoea_overrides:
            var_cfg["crossover"] = ("sbx", {"prob": 0.9, "eta": 15.0})
        if "mutation" not in agemoea_overrides:
            var_cfg["mutation"] = ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})

    builder = AGEMOEAConfig.builder()
    builder.pop_size(pop_size)

    c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
    builder.mutation(m_name, **m_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("agemoea")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


def build_rvea_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    rvea_variation: dict[str, Any] | None,
) -> tuple[Any, AlgorithmConfigProtocol]:
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    rvea_overrides = rvea_variation or {}
    var_cfg = resolve_default_variation_config(encoding, rvea_overrides)
    extra_cfg = {k: v for k, v in rvea_overrides.items() if k not in var_cfg}
    var_cfg.update(extra_cfg)

    if encoding == "real":
        if "crossover" not in rvea_overrides:
            var_cfg["crossover"] = ("sbx", {"prob": 1.0, "eta": 30.0})
        if "mutation" not in rvea_overrides:
            var_cfg["mutation"] = ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0})

    builder = RVEAConfig.builder()
    builder.pop_size(pop_size)
    builder.n_partitions(int(rvea_overrides.get("n_partitions", 12)))
    builder.alpha(float(rvea_overrides.get("alpha", 2.0)))
    if "adapt_freq" in rvea_overrides:
        builder.adapt_freq(rvea_overrides["adapt_freq"])
    else:
        builder.adapt_freq(0.1)

    c_name, c_kwargs = ensure_operator_tuple(var_cfg["crossover"], key="crossover")
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = ensure_operator_tuple(var_cfg["mutation"], key="mutation")
    builder.mutation(m_name, **m_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = ensure_operator_tuple(var_cfg["repair"], key="repair")
        builder.repair(ensure_supported_repair_name(encoding, r_name), **r_kwargs)

    cfg_data = cast(AlgorithmConfigProtocol, builder.build())
    algo_ctor = resolve_algorithm("rvea")
    cfg_dict = cfg_data.to_dict()
    return algo_ctor(cfg_dict, kernel), cfg_data


__all__ = [
    "build_nsgaii_algorithm",
    "build_moead_algorithm",
    "build_smsemoa_algorithm",
    "build_nsgaiii_algorithm",
    "build_spea2_algorithm",
    "build_ibea_algorithm",
    "build_smpso_algorithm",
    "build_agemoea_algorithm",
    "build_rvea_algorithm",
]

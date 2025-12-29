"""
Algorithm-specific builder helpers to keep the factory slim.
"""
from __future__ import annotations

from typing import Any, Tuple

from vamos.engine.algorithm.config import (
    NSGAIIConfig,
    MOEADConfig,
    SMSEMOAConfig,
    NSGAIIIConfig,
    SPEA2Config,
    IBEAConfig,
    SMPSOConfig,
)
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.config.variation import merge_variation_overrides, resolve_nsgaii_variation_config


def build_nsgaii_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    offspring_size: int,
    selection_pressure: int,
    external_archive_size: int | None,
    archive_type: str,
    nsgaii_variation: dict | None,
    track_genealogy: bool,
) -> Tuple[Any, Any]:
    encoding = getattr(problem, "encoding", "real")
    var_cfg = resolve_nsgaii_variation_config(encoding, nsgaii_variation)

    builder = NSGAIIConfig()
    builder.pop_size(pop_size)
    builder.offspring_size(offspring_size)
    builder.engine(engine_name)
    builder.survival("nsga2")

    if "crossover" in var_cfg:
        c_name, c_kwargs = var_cfg["crossover"]
        builder.crossover(c_name, **c_kwargs)
    if "mutation" in var_cfg:
        m_name, m_kwargs = var_cfg["mutation"]
        builder.mutation(m_name, **m_kwargs)
    if "selection" in var_cfg:
        s_name, s_kwargs = var_cfg["selection"]
        builder.selection(s_name, **s_kwargs)
    else:
        builder.selection("tournament", pressure=selection_pressure)

    if "repair" in var_cfg:
        r_name, r_kwargs = var_cfg["repair"]
        builder.repair(r_name, **r_kwargs)

    if "adaptive_operator_selection" in var_cfg:
        builder.adaptive_operator_selection(var_cfg["adaptive_operator_selection"])

    if external_archive_size:
        builder.external_archive(size=external_archive_size, archive_type=archive_type)
    if track_genealogy:
        builder.track_genealogy(True)

    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("nsgaii")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_moead_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    moead_variation: dict | None,
) -> Tuple[Any, Any]:
    var_cfg = merge_variation_overrides(
        {
            "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
            "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
            "aggregation": ("tchebycheff", {}),
        },
        moead_variation,
    )

    builder = MOEADConfig()
    builder.pop_size(pop_size)
    builder.engine(engine_name)
    builder.neighbor_size(20)
    builder.delta(0.9)
    builder.replace_limit(2)

    c_name, c_kwargs = var_cfg["crossover"]
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = var_cfg["mutation"]
    builder.mutation(m_name, **m_kwargs)

    a_name, a_kwargs = var_cfg["aggregation"]
    builder.aggregation(a_name, **a_kwargs)

    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("moead")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_smsemoa_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    smsemoa_variation: dict | None,
) -> Tuple[Any, Any]:
    var_cfg = merge_variation_overrides(
        {
            "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
            "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
            "selection": ("random", {}),
        },
        smsemoa_variation,
    )

    builder = SMSEMOAConfig()
    builder.pop_size(pop_size)
    builder.engine(engine_name)

    c_name, c_kwargs = var_cfg["crossover"]
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = var_cfg["mutation"]
    builder.mutation(m_name, **m_kwargs)

    s_name, s_kwargs = var_cfg["selection"]
    builder.selection(s_name, **s_kwargs)

    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("smsemoa")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_nsgaiii_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    nsgaiii_variation: dict | None,
    selection_pressure: int,
) -> Tuple[Any, Any]:
    var_cfg = merge_variation_overrides(
        {
            "crossover": ("sbx", {"prob": 1.0, "eta": 30.0}),
            "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
            "selection": ("tournament", {"pressure": selection_pressure}),
        },
        nsgaiii_variation,
    )

    builder = NSGAIIIConfig()
    builder.pop_size(pop_size)
    builder.engine(engine_name)

    c_name, c_kwargs = var_cfg["crossover"]
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = var_cfg["mutation"]
    builder.mutation(m_name, **m_kwargs)

    s_name, s_kwargs = var_cfg["selection"]
    builder.selection(s_name, **s_kwargs)

    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("nsgaiii")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_spea2_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    selection_pressure: int,
    external_archive_size: int | None,
    spea2_variation: dict | None,
) -> Tuple[Any, Any]:
    encoding = getattr(problem, "encoding", "real")
    spea2_overrides = spea2_variation or {}
    var_cfg = resolve_nsgaii_variation_config(encoding, spea2_overrides)
    extra_cfg = {k: v for k, v in spea2_overrides.items() if k not in {"crossover", "mutation", "repair"}}
    var_cfg.update(extra_cfg)

    builder = SPEA2Config()
    builder.pop_size(pop_size)
    archive_override = var_cfg.get("archive_size")
    builder.archive_size(int(archive_override) if archive_override is not None else (external_archive_size or pop_size))
    builder.engine(engine_name)
    if "k_neighbors" in var_cfg and var_cfg["k_neighbors"] is not None:
        builder.k_neighbors(int(var_cfg["k_neighbors"]))

    c_name, c_kwargs = var_cfg.get("crossover", ("sbx", {"prob": 0.9, "eta": 20.0}))
    builder.crossover(c_name, **c_kwargs)

    m_name, m_kwargs = var_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}))
    builder.mutation(m_name, **m_kwargs)

    sel_name, sel_kwargs = var_cfg.get("selection", ("tournament", {"pressure": selection_pressure}))
    builder.selection(sel_name, **sel_kwargs)

    if "repair" in var_cfg:
        r_name, r_kwargs = var_cfg["repair"]
        builder.repair(r_name, **r_kwargs)

    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("spea2")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_ibea_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    selection_pressure: int,
    ibea_variation: dict | None,
) -> Tuple[Any, Any]:
    encoding = getattr(problem, "encoding", "real")
    ibea_overrides = ibea_variation or {}
    var_cfg = resolve_nsgaii_variation_config(encoding, ibea_overrides)
    extra_cfg = {k: v for k, v in ibea_overrides.items() if k not in {"crossover", "mutation", "repair"}}
    var_cfg.update(extra_cfg)

    builder = IBEAConfig()
    builder.pop_size(pop_size)
    builder.engine(engine_name)
    c_name, c_kwargs = var_cfg.get("crossover", ("sbx", {"prob": 0.9, "eta": 20.0}))
    builder.crossover(c_name, **c_kwargs)
    m_name, m_kwargs = var_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}))
    builder.mutation(m_name, **m_kwargs)
    sel_name, sel_kwargs = var_cfg.get("selection", ("tournament", {"pressure": selection_pressure}))
    builder.selection(sel_name, **sel_kwargs)
    builder.indicator((var_cfg.get("indicator") or "eps"))
    builder.kappa(float(var_cfg.get("kappa", 0.05)))
    if "repair" in var_cfg:
        r_name, r_kwargs = var_cfg["repair"]
        builder.repair(r_name, **r_kwargs)
    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("ibea")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


def build_smpso_algorithm(
    *,
    kernel,
    engine_name: str,
    problem,
    pop_size: int,
    external_archive_size: int | None,
    smpso_variation: dict | None,
) -> Tuple[Any, Any]:
    mut_cfg = merge_variation_overrides(
        {
            "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
        },
        smpso_variation,
    )
    builder = SMPSOConfig()
    builder.pop_size(pop_size)
    builder.archive_size(pop_size if external_archive_size is None else external_archive_size)
    builder.engine(engine_name)
    m_name, m_kwargs = mut_cfg.get("mutation", ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}))
    builder.mutation(m_name, **m_kwargs)
    if "inertia" in mut_cfg:
        builder.inertia(float(mut_cfg["inertia"]))
    if "c1" in mut_cfg:
        builder.c1(float(mut_cfg["c1"]))
    if "c2" in mut_cfg:
        builder.c2(float(mut_cfg["c2"]))
    if "vmax_fraction" in mut_cfg:
        builder.vmax_fraction(float(mut_cfg["vmax_fraction"]))
    if "repair" in mut_cfg:
        r_name, r_kwargs = mut_cfg["repair"]
        builder.repair(r_name, **r_kwargs)
    cfg_data = builder.fixed()
    algo_ctor = resolve_algorithm("smpso")
    return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data


__all__ = [
    "build_nsgaii_algorithm",
    "build_moead_algorithm",
    "build_smsemoa_algorithm",
    "build_nsgaiii_algorithm",
    "build_spea2_algorithm",
    "build_ibea_algorithm",
    "build_smpso_algorithm",
]

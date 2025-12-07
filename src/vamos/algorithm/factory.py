from __future__ import annotations

from typing import Any

from vamos.algorithm.config import (
    NSGAIIConfig,
    MOEADConfig,
    SMSEMOAConfig,
    NSGAIIIConfig,
    SPEA2Config,
    IBEAConfig,
    SMPSOConfig,
)
from vamos.algorithm.autonsga2_builder import build_autonsga2
from vamos.algorithm.registry import resolve_algorithm
from vamos.experiment_config import ExperimentConfig
from vamos.config.variation import merge_variation_overrides, resolve_nsgaii_variation_config
from vamos.kernel.registry import resolve_kernel

def build_algorithm(
    algorithm_name: str,
    engine_name: str,
    problem,
    config: ExperimentConfig,
    *,
    external_archive_size: int | None = None,
    archive_type: str = "hypervolume",
    selection_pressure: int = 2,
    nsgaii_variation: dict | None = None,
    moead_variation: dict | None = None,
    smsemoa_variation: dict | None = None,
    nsga3_variation: dict | None = None,
    spea2_variation: dict | None = None,
    ibea_variation: dict | None = None,
    smpso_variation: dict | None = None,
    track_genealogy: bool = False,
):
    """
    Factory to build the algorithm instance.
    """
    kernel = resolve_kernel(engine_name)
    pop_size = config.population_size
    offspring_size = config.offspring_size()
    seed = config.seed

    # AutoNSGA-II special case (tuning)
    # If algorithm_name implies auto-tuning config structure, we might handle it differently.
    # But here we assume standard names.

    if algorithm_name == "nsgaii":
        # Resolve defaults based on problem encoding
        encoding = getattr(problem, "encoding", "real")
        var_cfg = resolve_nsgaii_variation_config(encoding, nsgaii_variation)

        builder = NSGAIIConfig()
        builder.pop_size(pop_size)
        builder.offspring_size(offspring_size)
        builder.engine(engine_name)
        builder.survival("nsga2")
        
        # Apply variation
        if "crossover" in var_cfg:
            c_name, c_kwargs = var_cfg["crossover"]
            builder.crossover(c_name, **c_kwargs)
        if "mutation" in var_cfg:
            m_name, m_kwargs = var_cfg["mutation"]
            # Adjust mutation prob if needed? Usually handled by operator or fixed value.
            # If prob is generic, we might want to scale by 1/n_var if not set.
            # For now assume explicit or default.
            builder.mutation(m_name, **m_kwargs)
        if "selection" in var_cfg:
            s_name, s_kwargs = var_cfg["selection"]
            builder.selection(s_name, **s_kwargs)
        else:
            builder.selection("tournament", pressure=selection_pressure)
        
        if "repair" in var_cfg:
            r_name, r_kwargs = var_cfg["repair"]
            builder.repair(r_name, **r_kwargs)

        if external_archive_size:
            builder.external_archive(size=external_archive_size, archive_type=archive_type)
        if track_genealogy:
            builder.track_genealogy(True)

        cfg_data = builder.fixed()
        algo_ctor = resolve_algorithm("nsgaii")
        return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data

    elif algorithm_name == "moead":
        # MOEA/D defaults
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
        builder.neighbor_size(20) # Default
        builder.delta(0.9)
        builder.replace_limit(2)
        
        c_name, c_kwargs = var_cfg["crossover"]
        builder.crossover(c_name, **c_kwargs)
        
        m_name, m_kwargs = var_cfg["mutation"]
        builder.mutation(m_name, **m_kwargs)
        
        a_name, a_kwargs = var_cfg["aggregation"]
        builder.aggregation(a_name, **a_kwargs)
        
        # Weight vectors?
        # builder.weight_vectors(...) # Default is usually decomposition based on obj

        cfg_data = builder.fixed()
        algo_ctor = resolve_algorithm("moead")
        return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data

    elif algorithm_name == "smsemoa":
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

    elif algorithm_name == "nsga3":
        var_cfg = merge_variation_overrides(
            {
                "crossover": ("sbx", {"prob": 1.0, "eta": 30.0}), # NSGA-III often uses higher eta
                "mutation": ("pm", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
                "selection": ("tournament", {"pressure": 2}),
            },
            nsga3_variation,
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
        
        # Reference directions?
        # builder.reference_directions(...)

        cfg_data = builder.fixed()
        algo_ctor = resolve_algorithm("nsga3")
        return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data

    elif algorithm_name == "spea2":
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

    elif algorithm_name == "ibea":
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

    elif algorithm_name == "smpso":
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

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

from __future__ import annotations

from typing import Any, Dict

from vamos.algorithm.config import (
    NSGAIIConfig,
    MOEADConfig,
    SMSEMOAConfig,
    NSGAIIIConfig,
)
from vamos.algorithm.autonsga2_builder import build_autonsga2
from vamos.algorithm.registry import resolve_algorithm
from vamos.experiment_config import ExperimentConfig
from vamos.kernel.registry import resolve_kernel

def resolve_nsgaii_variation_config(encoding: str, overrides: dict | None) -> dict:
    """
    Return a default variation config for NSGA-II based on encoding,
    optionally merging user overrides.
    """
    base = {}
    if encoding == "real":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),  # prob adjusted later usually
        }
    elif encoding == "binary":
        base = {
            "crossover": ("hux", {"prob": 0.9}),
            "mutation": ("bitflip", {"prob": 0.1}),
        }
    elif encoding == "integer":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}), # Integer SBX often used
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
        }
    elif encoding == "permutation":
        base = {
            "crossover": ("ox", {"prob": 0.9}),
            "mutation": ("swap", {"prob": 0.1}),
        }
    else:
        # Fallback
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
        }

    if overrides:
        # Simple merge: overrides replace base keys
        # For tuples like ("sbx", {...}), we replace the whole tuple if present
        if "crossover" in overrides:
            base["crossover"] = overrides["crossover"]
        if "mutation" in overrides:
            base["mutation"] = overrides["mutation"]
        if "repair" in overrides:
            base["repair"] = overrides["repair"]

    return base

def _merge_variation_overrides(base: dict | None, override: dict | None) -> dict:
    if base is None:
        base = {}
    if override is None:
        return base
    # Shallow merge
    return {**base, **override}

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

        cfg_data = builder.fixed()
        algo_ctor = resolve_algorithm("nsgaii")
        return algo_ctor(cfg_data.to_dict(), kernel=kernel), cfg_data

    elif algorithm_name == "moead":
        # MOEA/D defaults
        var_cfg = _merge_variation_overrides(
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
        var_cfg = _merge_variation_overrides(
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
        var_cfg = _merge_variation_overrides(
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

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

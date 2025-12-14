from __future__ import annotations

from typing import Any, Dict

import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.engine.algorithm.population import resolve_bounds
from vamos.engine.algorithm.selection import RandomSelection, TournamentSelection
from vamos.engine.algorithm.archive import CrowdingDistanceArchive, _single_front_crowding
from vamos.engine.operators.real import (
    BLXAlphaCrossover,
    SBXCrossover,
    PolynomialMutation,
    LinkedPolynomialMutation,
    UniformMutation,
    NonUniformMutation,
    LatinHypercubeInitializer,
    ScatterSearchInitializer,
    ClampRepair,
    ResampleRepair,
    RoundRepair,
)


def _repair_from_name(name: str):
    normalized = (name or "clip").lower()
    if normalized in {"clip", "clamp"}:
        return ClampRepair()
    if normalized in {"resample", "random"}:
        return ResampleRepair()
    if normalized == "round":
        return RoundRepair()
    return ClampRepair()


def build_autonsga2(config: Dict[str, Any], problem, seed: int) -> NSGAII:
    """
    Build an NSGA-II instance configured according to an AutoNSGA-II-like config dict.
    The expected keys follow the ParamSpace in the tuning example.
    """
    rng = np.random.default_rng(seed)
    pop_size = int(config.get("population_size", 100))
    offspring_size = int(config.get("offspring_size", pop_size))
    n_var = getattr(problem, "n_var", None) or getattr(problem, "n_variables", None)
    if n_var is None:
        raise ValueError("Problem must define n_var.")
    xl, xu = resolve_bounds(problem, getattr(problem, "encoding", "continuous"))

    # Initializer
    init_type = str(config.get("init.type", "random")).lower()
    if init_type == "lhs":
        initializer = LatinHypercubeInitializer(pop_size, xl, xu, rng=rng)
    elif init_type in {"scatter", "scatter_search"}:
        initializer = ScatterSearchInitializer(pop_size, xl, xu, base_size=max(20, pop_size // 2), rng=rng)
    else:
        initializer = None  # fall back to random in initialize_population

    # Selection
    sel_type = str(config.get("selection.type", "tournament")).lower()
    sel_size = int(config.get("selection.tournament_size", 2))
    if sel_type == "random":
        selection_cfg = ("random", {})
    else:
        selection_cfg = ("tournament", {"pressure": sel_size})

    # Repair
    repair_name = config.get("repair", "clip")
    repair_op = _repair_from_name(repair_name)

    # Crossover
    cx_type = str(config.get("crossover.type", "sbx")).lower()
    cx_prob = float(config.get("crossover.prob", 0.9))
    if cx_type == "sbx":
        eta = float(config.get("crossover.sbx_eta", 20.0))
        crossover = ("sbx", {"prob": cx_prob, "eta": eta})
    elif cx_type == "blx_alpha":
        alpha = float(config.get("crossover.blx_alpha", 0.5))
        blx_repair = config.get("crossover.blx_repair", repair_name)
        crossover = ("blx_alpha", {"prob": cx_prob, "alpha": alpha, "repair": blx_repair})
    elif cx_type == "arithmetic":
        crossover = ("arithmetic", {"prob": cx_prob})
    elif cx_type == "pcx":
        sigma_eta = float(config.get("crossover.pcx_sigma_eta", 0.1))
        sigma_zeta = float(config.get("crossover.pcx_sigma_zeta", 0.1))
        crossover = ("pcx", {"sigma_eta": sigma_eta, "sigma_zeta": sigma_zeta})
    elif cx_type == "undx":
        sigma_xi = float(config.get("crossover.undx_sigma_xi", 0.5))
        sigma_eta = float(config.get("crossover.undx_sigma_eta", 0.35))
        crossover = ("undx", {"sigma_xi": sigma_xi, "sigma_eta": sigma_eta})
    elif cx_type == "spx":
        epsilon = float(config.get("crossover.spx_epsilon", 0.5))
        crossover = ("spx", {"epsilon": epsilon})
    else:
        raise ValueError(f"Unsupported crossover.type '{cx_type}'")

    # Mutation
    mut_type = str(config.get("mutation.type", "polynomial")).lower()
    mut_factor = float(config.get("mutation.prob_factor", 1.0))
    prob_mut = mut_factor / float(max(1, n_var))
    if mut_type == "polynomial":
        eta = float(config.get("mutation.poly_eta", 20.0))
        mutation = ("pm", {"prob": prob_mut, "eta": eta})
    elif mut_type == "linked_polynomial":
        eta = float(config.get("mutation.poly_eta", 20.0))
        mutation = ("linked_polynomial", {"prob": prob_mut, "eta": eta})
    elif mut_type == "uniform":
        perturb = float(config.get("mutation.uniform_perturb", 0.1))
        mutation = ("uniform", {"prob": prob_mut, "perturb": perturb})
    elif mut_type == "non_uniform":
        perturb = float(config.get("mutation.non_uniform_perturb", 0.1))
        mutation = ("non_uniform", {"prob": prob_mut, "perturbation": perturb})
    elif mut_type == "gaussian":
        sigma = float(config.get("mutation.gaussian_sigma", 0.1))
        mutation = ("gaussian", {"prob": prob_mut, "sigma": sigma})
    elif mut_type == "uniform_reset":
        mutation = ("uniform_reset", {"prob": prob_mut})
    elif mut_type == "cauchy":
        gamma = float(config.get("mutation.cauchy_gamma", 0.1))
        mutation = ("cauchy", {"prob": prob_mut, "gamma": gamma})
    else:
        raise ValueError(f"Unsupported mutation.type '{mut_type}'")

    # Archive mode
    result_mode = str(config.get("result_mode", "population")).lower()
    archive_cfg = None
    if result_mode == "external_archive":
        archive_cfg = {"size": int(config.get("archive_size", 100))}

    # Build NSGA-II config
    cfg_builder = NSGAIIConfig()
    cfg_builder.pop_size(pop_size)
    cfg_builder.offspring_size(offspring_size)
    cfg_builder.crossover(crossover[0], **crossover[1])
    cfg_builder.mutation(mutation[0], **mutation[1])
    cfg_builder.selection(selection_cfg[0], **selection_cfg[1])
    cfg_builder.survival("nsga2")
    cfg_builder.engine("numpy")
    cfg_builder.initializer(init_type, **({"base_size": max(20, pop_size // 2)} if init_type == "scatter_search" else {}))
    cfg_builder.mutation_prob_factor(mut_factor)
    cfg_builder.result_mode(result_mode)
    if archive_cfg:
        cfg_builder.archive(archive_cfg["size"])

    cfg = cfg_builder.fixed().to_dict()
    algo = NSGAII(cfg, kernel=NumPyKernel())
    # Attach repair override for builders that expect it in cfg
    algo.cfg["repair"] = (repair_name, {})
    return algo


__all__ = ["build_autonsga2"]

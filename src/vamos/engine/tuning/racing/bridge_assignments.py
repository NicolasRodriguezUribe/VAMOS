"""
Assignment -> config builders for tuning.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.variants import ALGORITHM_VARIANT_GROUPS

_NSGAII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaii"]
_MOEAD_NAMES = ALGORITHM_VARIANT_GROUPS["moead"] - {"moead_permutation"}
_NSGAIII_NAMES = ALGORITHM_VARIANT_GROUPS["nsgaiii"]
_SMSEMOA_NAMES = ALGORITHM_VARIANT_GROUPS["smsemoa"]
_SPEA2_NAMES = ALGORITHM_VARIANT_GROUPS["spea2"]
_IBEA_NAMES = ALGORITHM_VARIANT_GROUPS["ibea"]
_SMPSO_NAMES = ALGORITHM_VARIANT_GROUPS["smpso"]
_AGEMOEA_NAMES = ALGORITHM_VARIANT_GROUPS["agemoea"]
_RVEA_NAMES = ALGORITHM_VARIANT_GROUPS["rvea"]
_TUNING_ARCHIVE_PRUNE_POLICIES = {"crowding", "hv", "mc_hv", "knn", "maxmin", "ref_dirs"}
_MIXED_CROSSOVER_NAMES = {"mixed", "uniform"}
_MIXED_MUTATION_NAMES = {"mixed", "gaussian"}
_MIXED_CROSSOVER_ASSIGNMENT_KEYS = (
    "perm_crossover",
    "perm_crossover_prob",
    "real_crossover",
    "real_crossover_prob",
    "int_crossover",
    "int_crossover_prob",
    "int_crossover_eta",
    "cat_crossover",
    "cat_crossover_prob",
)
_MIXED_MUTATION_ASSIGNMENT_KEYS = (
    "perm_mutation",
    "perm_mutation_prob",
    "real_mutation",
    "real_mutation_prob",
    "real_mutation_sigma",
    "real_mutation_sigma_factor",
    "real_mutation_eta",
    "int_mutation",
    "int_mutation_prob",
    "int_mutation_step",
    "int_mutation_eta",
    "cat_mutation",
    "cat_mutation_prob",
)
_MIXED_CROSSOVER_RATE_KEYS = (
    "perm_crossover_prob",
    "real_crossover_prob",
    "int_crossover_prob",
    "cat_crossover_prob",
)
_MIXED_MUTATION_RATE_KEYS = (
    "perm_mutation_prob",
    "real_mutation_prob",
    "int_mutation_prob",
    "cat_mutation_prob",
)


def _sanitize_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(assignment)
    use_external_archive = bool(sanitized.get("use_external_archive", False))
    archive_unbounded = bool(sanitized.get("archive_unbounded", False))
    if not use_external_archive or archive_unbounded:
        sanitized.pop("archive_prune_policy", None)
    if any(key in sanitized for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS):
        sanitized.setdefault("crossover", "mixed")
    if any(key in sanitized for key in _MIXED_MUTATION_ASSIGNMENT_KEYS):
        sanitized.setdefault("mutation", "mixed")
    if any(key in sanitized for key in _MIXED_CROSSOVER_RATE_KEYS):
        sanitized.pop("crossover_prob", None)
    if any(key in sanitized for key in _MIXED_MUTATION_RATE_KEYS):
        sanitized.pop("mutation_prob", None)
        sanitized.pop("mutation_prob_factor", None)
    return sanitized


def _apply_initializer(builder: Any, assignment: dict[str, Any], pop_size: int) -> None:
    initializer = str(assignment.get("initializer", "random")).strip().lower()
    if initializer == "random":
        builder.initializer("random")
    elif initializer == "lhs":
        builder.initializer("lhs")
    elif initializer == "scatter":
        factor = assignment.get("scatter_base_size_factor", 0.25)
        try:
            factor_f = float(factor)
        except (TypeError, ValueError):
            factor_f = 0.25
        factor_f = max(0.01, min(1.0, factor_f))
        base_size = int(math.floor(pop_size * factor_f + 0.5))
        base_size = max(2, min(pop_size, base_size))
        builder.initializer("scatter", base_size=base_size)
    elif initializer == "sobol":
        builder.initializer("sobol")
    elif initializer == "halton":
        builder.initializer("halton")
    elif initializer in {"obl", "opposition"}:
        builder.initializer("obl")


def _apply_optional_external_archive(builder: Any, assignment: dict[str, Any], pop_size: int) -> None:
    """Wire up external archive from tuning assignment (shared by all algorithms)."""
    use_external_archive = bool(assignment.get("use_external_archive", False))
    if not use_external_archive:
        return
    archive_unbounded = bool(assignment.get("archive_unbounded", False))
    if archive_unbounded:
        builder.external_archive(capacity=None)
        return
    prune_policy = str(assignment.get("archive_prune_policy", "crowding"))
    if prune_policy not in _TUNING_ARCHIVE_PRUNE_POLICIES:
        valid = ", ".join(sorted(_TUNING_ARCHIVE_PRUNE_POLICIES))
        raise ValueError(f"Unsupported prune_policy '{prune_policy}'. Expected one of: {valid}.")
    builder.external_archive(
        capacity=pop_size,
        pruning=prune_policy,
    )


def _apply_optional_repair(builder: Any, assignment: dict[str, Any]) -> None:
    repair = assignment.get("repair")
    if str(assignment.get("crossover", "")).strip().lower() == "blx_alpha" and assignment.get("blx_repair") is not None:
        repair = assignment.get("blx_repair")
    if repair is None:
        return
    repair_name = str(repair).strip().lower()
    if repair_name and repair_name not in {"none", "disabled", "false", "0"}:
        builder.repair(repair_name)


def _extend_real_crossover_params(cross: str, assignment: dict[str, Any], cross_params: dict[str, Any]) -> None:
    if cross == "sbx":
        cross_params["eta"] = float(assignment.get("crossover_eta", 20.0))
    elif cross == "blx_alpha":
        cross_params["alpha"] = float(assignment.get("crossover_alpha", 0.5))
    elif cross == "pcx":
        cross_params["sigma_eta"] = float(assignment.get("pcx_sigma_eta", 0.1))
        cross_params["sigma_zeta"] = float(assignment.get("pcx_sigma_zeta", 0.1))
    elif cross == "undx":
        cross_params["zeta"] = float(assignment.get("undx_zeta", 0.5))
        cross_params["eta"] = float(assignment.get("undx_eta", 0.35))
    elif cross == "simplex":
        cross_params["epsilon"] = float(assignment.get("simplex_epsilon", 0.5))
    elif cross == "blx_alpha_beta":
        cross_params["alpha"] = float(assignment.get("blxab_alpha", 0.75))
        cross_params["beta"] = float(assignment.get("blxab_beta", 0.25))
    elif cross == "whole_arithmetic":
        cross_params["alpha"] = float(assignment.get("wa_alpha", 0.5))
    elif cross == "laplace":
        cross_params["a"] = float(assignment.get("laplace_a", 0.0))
        cross_params["b"] = float(assignment.get("laplace_b", 0.5))
    elif cross == "fuzzy":
        cross_params["d"] = float(assignment.get("fuzzy_d", 0.5))


def _extend_mutation_params(mut: str, assignment: dict[str, Any], mut_params: dict[str, Any]) -> None:
    if mut in {"pm", "polynomial", "linked_polynomial"}:
        mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
    elif mut == "creep":
        mut_params["step"] = int(assignment.get("creep_step", 1))
    elif mut == "non_uniform":
        mut_params["perturbation"] = float(assignment.get("nonuniform_perturbation", 0.5))
    elif mut == "gaussian":
        mut_params["sigma"] = float(assignment.get("gaussian_sigma", 1.0))
    elif mut == "cauchy":
        mut_params["gamma"] = float(assignment.get("cauchy_gamma", 0.1))
    elif mut == "uniform":
        mut_params["perturb"] = float(assignment.get("uniform_perturb", 0.1))
    elif mut == "levy_flight":
        mut_params["beta"] = float(assignment.get("levy_beta", 1.5))
        mut_params["scale"] = float(assignment.get("levy_scale", 0.01))
    elif mut == "power_law":
        mut_params["index"] = float(assignment.get("power_index", 1.0))


def _has_mixed_crossover_assignment(assignment: dict[str, Any]) -> bool:
    return any(key in assignment for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS)


def _has_mixed_mutation_assignment(assignment: dict[str, Any]) -> bool:
    return any(key in assignment for key in _MIXED_MUTATION_ASSIGNMENT_KEYS)


def _extend_crossover_params(
    cross: str,
    assignment: dict[str, Any],
    cross_params: dict[str, Any],
    *,
    mixed: bool = False,
) -> None:
    normalized = str(cross).strip().lower()
    if mixed or _has_mixed_crossover_assignment(assignment):
        for key in _MIXED_CROSSOVER_ASSIGNMENT_KEYS:
            if key in assignment:
                cross_params[key] = assignment[key]
        return
    _extend_real_crossover_params(normalized, assignment, cross_params)


def _extend_operator_mutation_params(
    mut: str,
    assignment: dict[str, Any],
    mut_params: dict[str, Any],
    *,
    mixed: bool = False,
) -> None:
    normalized = str(mut).strip().lower()
    if mixed or _has_mixed_mutation_assignment(assignment):
        for key in _MIXED_MUTATION_ASSIGNMENT_KEYS:
            if key in assignment:
                mut_params[key] = assignment[key]
        return
    _extend_mutation_params(normalized, assignment, mut_params)


def _build_nsgaii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIConfig:
    builder = NSGAIIConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)

    _apply_initializer(builder, assignment, pop_size)

    raw_offspring_size = assignment.get("offspring_size")
    if raw_offspring_size is not None:
        offspring_size = int(raw_offspring_size)
    else:
        ratio = assignment.get("offspring_ratio", 1.0)
        try:
            ratio_f = float(ratio)
        except (TypeError, ValueError):
            ratio_f = 1.0
        ratio_f = max(0.0, min(1.0, ratio_f))
        offspring_size = int(math.floor(pop_size * ratio_f + 0.5))
    offspring_size = max(1, min(pop_size, int(offspring_size)))
    builder.offspring_size(offspring_size)

    cross = assignment["crossover"]
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(str(cross), assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = assignment["mutation"]
    mut_factor = assignment.get("mutation_prob_factor")
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    if mut_factor is not None:
        builder.mutation_prob_factor(float(mut_factor))
    _extend_operator_mutation_params(str(mut), assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    selection = str(assignment.get("selection", "tournament"))
    if selection == "tournament":
        builder.selection(selection, size=int(assignment.get("selection_size", 2)))
    else:
        builder.selection(selection)

    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)

    return builder.build()


def _build_moead_config(assignment: dict[str, Any], *, mixed: bool = False) -> MOEADConfig:
    builder = MOEADConfig.builder()
    builder.pop_size(int(assignment["pop_size"]))
    builder.neighbor_size(int(assignment["neighbor_size"]))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    cross = str(assignment["crossover"])
    if cross == "de":
        builder.crossover(
            "de",
            cr=float(assignment.get("de_cr", 1.0)),
            f=float(assignment.get("de_f", 0.5)),
        )
    else:
        cross_params: dict[str, Any] = {}
        if "crossover_prob" in assignment:
            cross_params["prob"] = float(assignment["crossover_prob"])
        _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
        builder.crossover(cross, **cross_params)

    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    aggregation = str(assignment["aggregation"])
    if aggregation == "pbi":
        builder.aggregation("pbi", theta=float(assignment.get("pbi_theta", 5.0)))
    else:
        builder.aggregation(aggregation)

    _apply_optional_external_archive(builder, assignment, int(assignment["pop_size"]))

    return builder.build()


def _build_moead_permutation_config(assignment: dict[str, Any]) -> MOEADConfig:
    builder = MOEADConfig.builder()
    builder.pop_size(int(assignment["pop_size"]))
    builder.neighbor_size(int(assignment["neighbor_size"]))
    builder.delta(float(assignment["delta"]))
    builder.replace_limit(int(assignment["replace_limit"]))
    builder.crossover(
        str(assignment["crossover"]),
        prob=float(assignment.get("crossover_prob", 0.9)),
    )
    builder.mutation(
        str(assignment["mutation"]),
        prob=assignment.get("mutation_prob", 0.1),
    )
    aggregation = str(assignment["aggregation"])
    if aggregation == "pbi":
        builder.aggregation("pbi", theta=float(assignment.get("pbi_theta", 5.0)))
    elif aggregation == "modified_tchebycheff":
        builder.aggregation("modified_tchebycheff", rho=float(assignment.get("mtch_rho", 0.001)))
    else:
        builder.aggregation(aggregation)

    _apply_optional_external_archive(builder, assignment, int(assignment["pop_size"]))

    return builder.build()


def _build_nsgaiii_config(assignment: dict[str, Any], *, mixed: bool = False) -> NSGAIIIConfig:
    builder = NSGAIIIConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)
    _apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def _build_smsemoa_config(assignment: dict[str, Any], *, mixed: bool = False) -> SMSEMOAConfig:
    builder = SMSEMOAConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)
    _apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.reference_point(offset=0.1, adaptive=True)
    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def _build_spea2_config(assignment: dict[str, Any], *, mixed: bool = False) -> SPEA2Config:
    builder = SPEA2Config.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)
    builder.archive_size(int(assignment.get("archive_size", pop_size)))
    _apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.k_neighbors(int(assignment.get("k_neighbors", max(1, int(np.sqrt(pop_size))))))
    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def _build_ibea_config(assignment: dict[str, Any], *, mixed: bool = False) -> IBEAConfig:
    builder = IBEAConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)
    _apply_initializer(builder, assignment, pop_size)
    cross = str(assignment["crossover"])
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)
    mut = str(assignment["mutation"])
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)
    builder.selection("tournament", size=int(assignment["selection_pressure"]))
    builder.indicator(str(assignment.get("indicator", "eps")))
    builder.kappa(float(assignment.get("kappa", 1.0)))
    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def _build_smpso_config(assignment: dict[str, Any], *, mixed: bool = False) -> SMPSOConfig:
    builder = SMPSOConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)
    builder.archive_size(int(assignment.get("archive_size", pop_size)))
    builder.inertia(float(assignment["inertia"]))
    builder.c1(float(assignment["c1"]))
    builder.c2(float(assignment["c2"]))
    builder.vmax_fraction(float(assignment["vmax_fraction"]))
    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = float(assignment["mutation_prob"])
    if mut in {"pm", "polynomial"}:
        mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
    elif mixed or _has_mixed_mutation_assignment(assignment):
        _extend_operator_mutation_params(mut, assignment, mut_params, mixed=True)
    builder.mutation(mut, **mut_params)
    _apply_optional_external_archive(builder, assignment, pop_size)
    return builder.build()


def _build_agemoea_config(assignment: dict[str, Any], *, mixed: bool = False) -> AGEMOEAConfig:
    builder = AGEMOEAConfig.builder()
    pop_size = int(assignment["pop_size"])
    builder.pop_size(pop_size)

    _apply_initializer(builder, assignment, pop_size)

    cross = str(assignment.get("crossover", "sbx"))
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)

    return builder.build()


def _build_rvea_config(assignment: dict[str, Any], *, mixed: bool = False) -> RVEAConfig:
    builder = RVEAConfig.builder()
    n_partitions = int(assignment.get("n_partitions", 12))
    n_obj = max(2, int(assignment.get("n_obj", 2)))
    pop_size = int(math.comb(n_partitions + n_obj - 1, n_obj - 1))
    builder.pop_size(pop_size)
    builder.n_partitions(n_partitions)
    builder.alpha(float(assignment.get("alpha", 2.0)))
    builder.adapt_freq(float(assignment.get("adapt_freq", 0.1)))

    _apply_initializer(builder, assignment, pop_size)

    cross = str(assignment.get("crossover", "sbx"))
    cross_params: dict[str, Any] = {}
    if "crossover_prob" in assignment:
        cross_params["prob"] = float(assignment["crossover_prob"])
    _extend_crossover_params(cross, assignment, cross_params, mixed=mixed)
    builder.crossover(cross, **cross_params)

    mut = str(assignment.get("mutation", "pm"))
    mut_params: dict[str, Any] = {}
    if "mutation_prob" in assignment:
        mut_params["prob"] = assignment["mutation_prob"]
    _extend_operator_mutation_params(mut, assignment, mut_params, mixed=mixed)
    builder.mutation(mut, **mut_params)

    _apply_optional_repair(builder, assignment)
    _apply_optional_external_archive(builder, assignment, pop_size)

    return builder.build()


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> AlgorithmConfigProtocol:
    """
    Build a concrete algorithm config dataclass from a sampled assignment.
    """

    algo = algorithm_name.lower()
    assignment = _sanitize_assignment(assignment)
    if algo == "nsgaii_mixed":
        return _build_nsgaii_config(assignment, mixed=True)
    if algo in _NSGAII_NAMES:
        return _build_nsgaii_config(assignment)
    if algo == "moead_mixed":
        return _build_moead_config(assignment, mixed=True)
    if algo in _MOEAD_NAMES:
        return _build_moead_config(assignment)
    if algo == "moead_permutation":
        return _build_moead_permutation_config(assignment)
    if algo == "nsgaiii_mixed":
        return _build_nsgaiii_config(assignment, mixed=True)
    if algo in _NSGAIII_NAMES:
        return _build_nsgaiii_config(assignment)
    if algo == "smsemoa_mixed":
        return _build_smsemoa_config(assignment, mixed=True)
    if algo in _SMSEMOA_NAMES:
        return _build_smsemoa_config(assignment)
    if algo == "spea2_mixed":
        return _build_spea2_config(assignment, mixed=True)
    if algo in _SPEA2_NAMES:
        return _build_spea2_config(assignment)
    if algo == "ibea_mixed":
        return _build_ibea_config(assignment, mixed=True)
    if algo in _IBEA_NAMES:
        return _build_ibea_config(assignment)
    if algo == "smpso_mixed":
        return _build_smpso_config(assignment, mixed=True)
    if algo in _SMPSO_NAMES:
        return _build_smpso_config(assignment)
    if algo == "agemoea_mixed":
        return _build_agemoea_config(assignment, mixed=True)
    if algo in _AGEMOEA_NAMES:
        return _build_agemoea_config(assignment)
    if algo == "rvea_mixed":
        return _build_rvea_config(assignment, mixed=True)
    if algo in _RVEA_NAMES:
        return _build_rvea_config(assignment)
    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")


__all__ = ["config_from_assignment"]

"""
Bridge from sampled hyperparameters to concrete algorithm configs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.engine.algorithm.config import (
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMSEMOAConfig,
    SPEA2Config,
    IBEAConfig,
    SMPSOConfig,
)
from .config_space import AlgorithmConfigSpace
from .param_space import (
    Real,
    Int,
    Categorical,
    Boolean,
    Condition,
    ParamType,
    ConditionalBlock,
)


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical(
            "offspring_ratio",
            [0.25, 0.5, 0.75, 1.0],
        ),
        Categorical("initializer", ["random", "lhs", "scatter"]),
        Categorical("crossover", ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["pm", "linked_polynomial", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform"]),
        Real("mutation_prob_factor", 0.25, 3.0),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("selection", ["tournament"]),
        Int("selection_pressure", 2, 10),
        Categorical("repair", ["none", "clip", "reflect", "random", "round"]),
        Boolean("use_external_archive"),
        Boolean("archive_unbounded"),
    ]
    archive_type_param = Categorical("archive_type", ["hypervolume", "crowding"])
    archive_size_factor_param = Categorical("archive_size_factor", [1, 2, 5, 10])
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock(
            "crossover",
            "blx_alpha",
            [
                Real("crossover_alpha", 0.0, 1.0),
                Categorical("blx_repair", ["clip", "random", "reflect", "round"]),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "pcx",
            [
                Real("pcx_sigma_eta", 0.01, 0.5),
                Real("pcx_sigma_zeta", 0.01, 0.5),
            ],
        ),
        ConditionalBlock(
            "crossover",
            "undx",
            [
                Real("undx_zeta", 0.1, 1.0),
                Real("undx_eta", 0.1, 1.0),
            ],
        ),
        ConditionalBlock("crossover", "simplex", [Real("simplex_epsilon", 0.1, 1.0)]),
        ConditionalBlock("mutation", "non_uniform", [Real("nonuniform_perturbation", 0.05, 0.5)]),
        ConditionalBlock("mutation", "gaussian", [Real("gaussian_sigma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "cauchy", [Real("cauchy_gamma", 0.001, 0.5)]),
        ConditionalBlock("mutation", "uniform", [Real("uniform_perturb", 0.01, 0.5)]),
        ConditionalBlock(
            "initializer",
            "scatter",
            [
                Categorical("scatter_base_size_factor", [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]),
            ],
        ),
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                archive_type_param,
                archive_size_factor_param,
            ],
        ),
    ]
    conditions = [
        Condition("archive_type", "cfg['archive_unbounded'] == False"),
        Condition("archive_size_factor", "cfg['archive_unbounded'] == False"),
    ]
    return AlgorithmConfigSpace("nsgaii", params, conditionals, conditions)


def build_moead_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("neighbor_size", 5, 40),
        Real("delta", 0.5, 0.95),
        Int("replace_limit", 1, 5),
        Categorical("crossover", ["sbx", "de"]),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("aggregation", ["tchebycheff", "weighted_sum", "pbi"]),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_prob", 0.6, 1.0), Real("crossover_eta", 10.0, 40.0)]),
        ConditionalBlock("crossover", "de", [Real("de_cr", 0.0, 1.0), Real("de_f", 0.0, 1.0)]),
        ConditionalBlock("aggregation", "pbi", [Real("pbi_theta", 1.0, 10.0)]),
    ]
    return AlgorithmConfigSpace("moead", params, conditionals)


def build_nsgaiii_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Real("crossover_eta", 20.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
    ]
    return AlgorithmConfigSpace("nsgaiii", params, [])


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 1.0),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
    ]
    return AlgorithmConfigSpace("smsemoa", params, [])


def build_spea2_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 0.95),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
        Int("k_neighbors", 1, 25),
    ]
    return AlgorithmConfigSpace("spea2", params, [])


def build_ibea_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Categorical("crossover", ["sbx"]),
        Real("crossover_prob", 0.6, 0.95),
        Real("crossover_eta", 10.0, 40.0),
        Categorical("mutation", ["pm"]),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
        Int("selection_pressure", 2, 10),
        Categorical("indicator", ["eps", "hypervolume"]),
        Real("kappa", 0.01, 0.2),
    ]
    return AlgorithmConfigSpace("ibea", params, [])


def build_smpso_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("archive_size", 20, 200, log=True),
        Real("inertia", 0.1, 0.9),
        Real("c1", 0.5, 2.5),
        Real("c2", 0.5, 2.5),
        Real("vmax_fraction", 0.1, 1.0),
        Real("mutation_prob", 0.01, 0.5),
        Real("mutation_eta", 5.0, 40.0),
    ]
    return AlgorithmConfigSpace("smpso", params, [])


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> Any:
    """
    Build a concrete algorithm config dataclass from a sampled assignment.
    """
    builder: Any
    algo = algorithm_name.lower()
    if algo == "nsgaii":
        builder = NSGAIIConfig.builder()
        pop_size = int(assignment["pop_size"])
        builder.pop_size(pop_size)

        initializer = str(assignment.get("initializer", "random")).strip().lower()
        if initializer == "lhs":
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
        cross_params = {"prob": float(assignment["crossover_prob"])}
        if cross == "sbx":
            cross_params["eta"] = float(assignment.get("crossover_eta", 20.0))
        elif cross == "blx_alpha":
            cross_params["alpha"] = float(assignment.get("crossover_alpha", 0.5))
            blx_repair = assignment.get("blx_repair")
            if blx_repair is not None:
                cross_params["repair"] = str(blx_repair)
        elif cross == "pcx":
            cross_params["sigma_eta"] = float(assignment.get("pcx_sigma_eta", 0.1))
            cross_params["sigma_zeta"] = float(assignment.get("pcx_sigma_zeta", 0.1))
        elif cross == "undx":
            cross_params["zeta"] = float(assignment.get("undx_zeta", 0.5))
            cross_params["eta"] = float(assignment.get("undx_eta", 0.35))
        elif cross == "simplex":
            cross_params["epsilon"] = float(assignment.get("simplex_epsilon", 0.5))
        builder.crossover(cross, **cross_params)
        mut = assignment["mutation"]
        mut_factor = assignment.get("mutation_prob_factor")
        mut_params = {"prob": float(assignment.get("mutation_prob", 0.1))}
        if mut_factor is not None:
            builder.mutation_prob_factor(float(mut_factor))
        if mut in {"pm", "polynomial", "linked_polynomial"}:
            mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
        elif mut == "non_uniform":
            mut_params["perturbation"] = float(assignment.get("nonuniform_perturbation", 0.5))
        elif mut == "gaussian":
            mut_params["sigma"] = float(assignment.get("gaussian_sigma", 0.1))
        elif mut == "cauchy":
            mut_params["gamma"] = float(assignment.get("cauchy_gamma", 0.1))
        elif mut == "uniform":
            mut_params["perturb"] = float(assignment.get("uniform_perturb", 0.1))
        builder.mutation(mut, **mut_params)
        builder.selection(str(assignment.get("selection", "tournament")), pressure=int(assignment["selection_pressure"]))

        repair = assignment.get("repair")
        if repair is not None:
            repair_name = str(repair).strip().lower()
            if repair_name and repair_name not in {"none", "disabled", "false", "0"}:
                builder.repair(repair_name)

        use_external_archive = bool(assignment.get("use_external_archive", False))
        if use_external_archive:
            archive_type_raw = assignment.get("archive_type", "hypervolume")
            archive_unbounded = bool(assignment.get("archive_unbounded", False))
            if str(archive_type_raw).strip().lower() == "unbounded":
                archive_unbounded = True
            if archive_unbounded:
                builder.archive(int(assignment["pop_size"]), unbounded=True)
            else:
                archive_type = str(archive_type_raw)
                archive_size_factor = int(assignment.get("archive_size_factor", 1))
                if archive_size_factor < 1:
                    raise ValueError("archive_size_factor must be >= 1.")
                pop_size = int(assignment["pop_size"])
                archive_size = max(pop_size, pop_size * archive_size_factor)
                builder.archive_type(archive_type)
                builder.archive(archive_size)
        return builder.build()
    if algo == "moead":
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
            builder.crossover(
                "sbx",
                prob=float(assignment.get("crossover_prob", 1.0)),
                eta=float(assignment.get("crossover_eta", 20.0)),
            )
        builder.mutation(
            str(assignment["mutation"]),
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        aggregation = str(assignment["aggregation"])
        if aggregation == "pbi":
            builder.aggregation("pbi", theta=float(assignment.get("pbi_theta", 5.0)))
        else:
            builder.aggregation(aggregation)
        return builder.build()
    if algo == "nsgaiii":
        builder = NSGAIIIConfig.builder()
        builder.pop_size(int(assignment["pop_size"]))
        builder.crossover(
            str(assignment["crossover"]),
            prob=float(assignment["crossover_prob"]),
            eta=float(assignment["crossover_eta"]),
        )
        builder.mutation(
            str(assignment["mutation"]),
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        builder.selection("tournament", pressure=int(assignment["selection_pressure"]))
        return builder.build()
    if algo == "smsemoa":
        builder = SMSEMOAConfig.builder()
        builder.pop_size(int(assignment["pop_size"]))
        builder.crossover(
            str(assignment["crossover"]),
            prob=float(assignment["crossover_prob"]),
            eta=float(assignment["crossover_eta"]),
        )
        builder.mutation(
            str(assignment["mutation"]),
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        builder.selection("tournament", pressure=int(assignment["selection_pressure"]))
        builder.reference_point(offset=0.1, adaptive=True)
        return builder.build()
    if algo == "spea2":
        builder = SPEA2Config.builder()
        builder.pop_size(int(assignment["pop_size"]))
        builder.archive_size(int(assignment.get("archive_size", assignment["pop_size"])))
        builder.crossover(
            str(assignment["crossover"]),
            prob=float(assignment["crossover_prob"]),
            eta=float(assignment.get("crossover_eta", 20.0)),
        )
        builder.mutation(
            str(assignment["mutation"]),
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        builder.selection("tournament", pressure=int(assignment["selection_pressure"]))
        builder.k_neighbors(int(assignment.get("k_neighbors", max(1, int(np.sqrt(assignment["pop_size"]))))))
        return builder.build()
    if algo == "ibea":
        builder = IBEAConfig.builder()
        builder.pop_size(int(assignment["pop_size"]))
        builder.crossover(
            str(assignment["crossover"]),
            prob=float(assignment["crossover_prob"]),
            eta=float(assignment.get("crossover_eta", 20.0)),
        )
        builder.mutation(
            str(assignment["mutation"]),
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        builder.selection("tournament", pressure=int(assignment["selection_pressure"]))
        builder.indicator(str(assignment.get("indicator", "eps")))
        builder.kappa(float(assignment.get("kappa", 1.0)))
        return builder.build()
    if algo == "smpso":
        builder = SMPSOConfig.builder()
        builder.pop_size(int(assignment["pop_size"]))
        builder.archive_size(int(assignment.get("archive_size", assignment["pop_size"])))
        builder.inertia(float(assignment["inertia"]))
        builder.c1(float(assignment["c1"]))
        builder.c2(float(assignment["c2"]))
        builder.vmax_fraction(float(assignment["vmax_fraction"]))
        builder.mutation(
            "pm",
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        return builder.build()
    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")

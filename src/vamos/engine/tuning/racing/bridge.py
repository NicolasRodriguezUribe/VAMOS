"""
Bridge from sampled hyperparameters to concrete algorithm configs.
"""

from __future__ import annotations

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
    ParamType,
    ConditionalBlock,
)


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    params: list[ParamType] = [
        Int("pop_size", 20, 200, log=True),
        Int("offspring_size", 20, 200, log=True),
        Categorical("crossover", ["sbx", "blx_alpha"]),
        Real("crossover_prob", 0.6, 1.0),
        Categorical("mutation", ["pm", "non_uniform"]),
        Real("mutation_prob_factor", 0.5, 3.0),
        Real("mutation_eta", 5.0, 40.0),
        Categorical("selection", ["tournament"]),
        Int("selection_pressure", 2, 4),
        Boolean("use_external_archive"),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [Real("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("crossover", "blx_alpha", [Real("crossover_alpha", 0.1, 0.8)]),
        ConditionalBlock(
            "use_external_archive",
            True,
            [
                Categorical("archive_type", ["hypervolume", "crowding", "unbounded"]),
                Categorical("archive_size_factor", [1, 2, 5, 10]),
            ],
        ),
    ]
    return AlgorithmConfigSpace("nsgaii", params, conditionals)


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
        Int("selection_pressure", 2, 4),
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
        Int("selection_pressure", 2, 4),
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
        Int("selection_pressure", 2, 4),
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
        Int("selection_pressure", 2, 4),
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
        builder.pop_size(int(assignment["pop_size"]))
        builder.offspring_size(int(assignment.get("offspring_size", assignment["pop_size"])))
        cross = assignment["crossover"]
        cross_params = {"prob": float(assignment["crossover_prob"])}
        if cross == "sbx":
            cross_params["eta"] = float(assignment.get("crossover_eta", 20.0))
        elif cross == "blx_alpha":
            cross_params["alpha"] = float(assignment.get("crossover_alpha", 0.5))
        builder.crossover(cross, **cross_params)
        mut = assignment["mutation"]
        mut_factor = assignment.get("mutation_prob_factor")
        mut_params = {"prob": float(assignment.get("mutation_prob", 0.1))}
        if mut_factor is not None:
            builder.mutation_prob_factor(float(mut_factor))
        if mut == "pm":
            mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
        builder.mutation(mut, **mut_params)
        builder.selection(str(assignment.get("selection", "tournament")), pressure=int(assignment["selection_pressure"]))

        use_external_archive = bool(assignment.get("use_external_archive", False))
        if use_external_archive:
            archive_type = str(assignment.get("archive_type", "hypervolume"))
            if archive_type == "unbounded":
                builder.archive(int(assignment["pop_size"]), unbounded=True)
            else:
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

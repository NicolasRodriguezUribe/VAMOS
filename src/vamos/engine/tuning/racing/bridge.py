"""
Bridge from sampled hyperparameters to concrete algorithm configs.
"""
from __future__ import annotations

from typing import Any, Dict

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
from .parameters import (
    CategoricalParam,
    ConditionalBlock,
    FloatParam,
    IntegerParam,
)


def build_nsgaii_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        IntegerParam("offspring_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        CategoricalParam("crossover", ["sbx", "blx_alpha"]),
        FloatParam("crossover_prob", 0.6, 0.95),
        CategoricalParam("mutation", ["pm", "non_uniform"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        CategoricalParam("selection", ["tournament"]),
        IntegerParam("selection_pressure", 2, 4),
    ]
    conditionals = [
        ConditionalBlock("crossover", "sbx", [FloatParam("crossover_eta", 5.0, 40.0)]),
        ConditionalBlock("crossover", "blx_alpha", [FloatParam("crossover_alpha", 0.1, 0.8)]),
    ]
    return AlgorithmConfigSpace("nsgaii", params, conditionals)


def build_moead_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        IntegerParam("neighbor_size", 5, 40, log=False),
        FloatParam("delta", 0.5, 0.95),
        IntegerParam("replace_limit", 1, 5),
        CategoricalParam("crossover", ["sbx"]),
        FloatParam("crossover_prob", 0.6, 1.0),
        FloatParam("crossover_eta", 10.0, 40.0),
        CategoricalParam("mutation", ["pm"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        CategoricalParam("aggregation", ["tchebycheff", "weighted_sum"]),
    ]
    return AlgorithmConfigSpace("moead", params, [])


def build_nsgaiii_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        CategoricalParam("crossover", ["sbx"]),
        FloatParam("crossover_prob", 0.6, 1.0),
        FloatParam("crossover_eta", 20.0, 40.0),
        CategoricalParam("mutation", ["pm"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        IntegerParam("selection_pressure", 2, 4),
    ]
    return AlgorithmConfigSpace("nsgaiii", params, [])


def build_smsemoa_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        CategoricalParam("crossover", ["sbx"]),
        FloatParam("crossover_prob", 0.6, 1.0),
        FloatParam("crossover_eta", 10.0, 40.0),
        CategoricalParam("mutation", ["pm"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        IntegerParam("selection_pressure", 2, 4),
    ]
    return AlgorithmConfigSpace("smsemoa", params, [])


def build_spea2_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        IntegerParam("archive_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        CategoricalParam("crossover", ["sbx"]),
        FloatParam("crossover_prob", 0.6, 0.95),
        FloatParam("crossover_eta", 10.0, 40.0),
        CategoricalParam("mutation", ["pm"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        IntegerParam("selection_pressure", 2, 4),
        IntegerParam("k_neighbors", 1, 25),
    ]
    return AlgorithmConfigSpace("spea2", params, [])


def build_ibea_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        CategoricalParam("crossover", ["sbx"]),
        FloatParam("crossover_prob", 0.6, 0.95),
        FloatParam("crossover_eta", 10.0, 40.0),
        CategoricalParam("mutation", ["pm"]),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
        IntegerParam("selection_pressure", 2, 4),
        CategoricalParam("indicator", ["eps", "hypervolume"]),
        FloatParam("kappa", 0.01, 0.2),
    ]
    return AlgorithmConfigSpace("ibea", params, [])


def build_smpso_config_space() -> AlgorithmConfigSpace:
    params = [
        IntegerParam("pop_size", 20, 200, log=True),
        IntegerParam("archive_size", 20, 200, log=True),
        CategoricalParam("engine", ["numpy", "numba", "moocore"]),
        FloatParam("inertia", 0.1, 0.9),
        FloatParam("c1", 0.5, 2.5),
        FloatParam("c2", 0.5, 2.5),
        FloatParam("vmax_fraction", 0.1, 1.0),
        FloatParam("mutation_prob", 0.01, 0.5),
        FloatParam("mutation_eta", 5.0, 40.0),
    ]
    return AlgorithmConfigSpace("smpso", params, [])


def config_from_assignment(algorithm_name: str, assignment: Dict[str, Any]):
    """
    Build a concrete algorithm config dataclass from a sampled assignment.
    """
    algo = algorithm_name.lower()
    if algo == "nsgaii":
        builder = NSGAIIConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.offspring_size(int(assignment.get("offspring_size", assignment["pop_size"])))
        builder.engine(str(assignment.get("engine", "numpy")))
        cross = assignment["crossover"]
        cross_params = {"prob": float(assignment["crossover_prob"])}
        if cross == "sbx":
            cross_params["eta"] = float(assignment.get("crossover_eta", 20.0))
        elif cross == "blx_alpha":
            cross_params["alpha"] = float(assignment.get("crossover_alpha", 0.5))
        builder.crossover(cross, **cross_params)
        mut = assignment["mutation"]
        mut_params = {"prob": float(assignment["mutation_prob"])}
        if mut == "pm":
            mut_params["eta"] = float(assignment.get("mutation_eta", 20.0))
        builder.mutation(mut, **mut_params)
        builder.selection(str(assignment.get("selection", "tournament")), pressure=int(assignment["selection_pressure"]))
        builder.survival("nsga2")
        return builder.fixed()
    if algo == "moead":
        builder = MOEADConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.engine(str(assignment.get("engine", "numpy")))
        builder.neighbor_size(int(assignment["neighbor_size"]))
        builder.delta(float(assignment["delta"]))
        builder.replace_limit(int(assignment["replace_limit"]))
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
        builder.aggregation(str(assignment["aggregation"]))
        return builder.fixed()
    if algo == "nsgaiii":
        builder = NSGAIIIConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.engine(str(assignment.get("engine", "numpy")))
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
        return builder.fixed()
    if algo == "smsemoa":
        builder = SMSEMOAConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.engine(str(assignment.get("engine", "numpy")))
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
        return builder.fixed()
    if algo == "spea2":
        builder = SPEA2Config()
        builder.pop_size(int(assignment["pop_size"]))
        builder.archive_size(int(assignment.get("archive_size", assignment["pop_size"])))
        builder.engine(str(assignment.get("engine", "numpy")))
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
        return builder.fixed()
    if algo == "ibea":
        builder = IBEAConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.engine(str(assignment.get("engine", "numpy")))
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
        builder.kappa(float(assignment.get("kappa", 0.05)))
        return builder.fixed()
    if algo == "smpso":
        builder = SMPSOConfig()
        builder.pop_size(int(assignment["pop_size"]))
        builder.archive_size(int(assignment.get("archive_size", assignment["pop_size"])))
        builder.engine(str(assignment.get("engine", "numpy")))
        builder.inertia(float(assignment["inertia"]))
        builder.c1(float(assignment["c1"]))
        builder.c2(float(assignment["c2"]))
        builder.vmax_fraction(float(assignment["vmax_fraction"]))
        builder.mutation(
            "pm",
            prob=float(assignment["mutation_prob"]),
            eta=float(assignment["mutation_eta"]),
        )
        return builder.fixed()
    raise ValueError(f"Unsupported algorithm for config construction: {algorithm_name}")

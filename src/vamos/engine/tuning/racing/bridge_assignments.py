"""
Assignment -> config builders for tuning.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.engine.algorithm.config import IBEAConfig, MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SMPSOConfig, SMSEMOAConfig, SPEA2Config


def config_from_assignment(algorithm_name: str, assignment: dict[str, Any]) -> Any:
    """
    Build a concrete algorithm config dataclass from a sampled assignment.
    """
    builder: Any
    algo = algorithm_name.lower()
    if algo in {"nsgaii", "nsgaii_permutation", "nsgaii_mixed"}:
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
        cross_params: dict[str, Any] = {"prob": float(assignment["crossover_prob"])}
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
        mut_params: dict[str, Any] = {"prob": assignment.get("mutation_prob", 0.1)}
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
    if algo == "moead_permutation":
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

        use_external_archive = bool(assignment.get("use_external_archive", False))
        if use_external_archive:
            archive_type = str(assignment.get("archive_type", "crowding"))
            archive_size_factor = int(assignment.get("archive_size_factor", 1))
            if archive_size_factor < 1:
                raise ValueError("archive_size_factor must be >= 1.")
            pop_size = int(assignment["pop_size"])
            archive_size = max(pop_size, pop_size * archive_size_factor)
            builder.archive_type(archive_type)
            builder.archive(archive_size)
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


__all__ = ["config_from_assignment"]

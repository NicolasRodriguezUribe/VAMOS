from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.config import (
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol


def resolve_problem(
    problem: str | ProblemProtocol,
    *,
    n_var: int | None = None,
    n_obj: int | None = None,
    **problem_kwargs: Any,
) -> ProblemProtocol:
    """Resolve problem from string name or return as-is if already a Problem."""
    if isinstance(problem, str):
        kwargs = dict(problem_kwargs)
        if n_var is not None:
            kwargs["n_var"] = n_var
        if n_obj is not None:
            kwargs["n_obj"] = n_obj
        selection = make_problem_selection(problem, **kwargs)
        return selection.instantiate()
    return problem


def build_nsgaii_config(
    *,
    pop_size: int,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    engine: str,
    selection_pressure: int = 2,
    archive_size: int | None = None,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    offspring_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
) -> dict[str, Any]:
    """Build NSGA-II config dict with all available parameters."""
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .survival("nsga2")
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if offspring_size is not None:
        cfg = cfg.offspring_size(offspring_size)
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    return cfg.fixed().to_dict()


def build_moead_config(
    *,
    pop_size: int,
    neighbor_size: int,
    delta: float,
    replace_limit: int,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    aggregation: str,
    engine: str,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    archive_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
) -> dict[str, Any]:
    """Build MOEA/D config dict with all available parameters."""
    cfg = (
        MOEADConfig()
        .pop_size(pop_size)
        .neighbor_size(neighbor_size)
        .delta(delta)
        .replace_limit(replace_limit)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .aggregation(aggregation)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    return cfg.fixed().to_dict()


def build_spea2_config(
    *,
    pop_size: int,
    archive_size: int | None,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    selection_pressure: int = 2,
    k_neighbors: int | None = None,
    engine: str,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    result_mode: str = "non_dominated",
) -> dict[str, Any]:
    """Build SPEA2 config dict with all available parameters."""
    cfg = (
        SPEA2Config()
        .pop_size(pop_size)
        .archive_size(archive_size or pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
    )
    if k_neighbors is not None:
        cfg = cfg.k_neighbors(k_neighbors)
    return cfg.fixed().to_dict()


def build_smsemoa_config(
    *,
    pop_size: int,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    selection_pressure: int = 2,
    ref_point_offset: float = 0.1,
    ref_point_adaptive: bool = True,
    engine: str,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    archive_size: int | None = None,
    result_mode: str = "non_dominated",
    archive_type: str = "hypervolume",
) -> dict[str, Any]:
    """Build SMS-EMOA config dict with all available parameters."""
    cfg = (
        SMSEMOAConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .reference_point(offset=ref_point_offset, adaptive=ref_point_adaptive)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
        .archive_type(archive_type)
    )
    if archive_size is not None and archive_size > 0:
        cfg = cfg.external_archive(size=archive_size, archive_type=archive_type)
    return cfg.fixed().to_dict()


def build_nsgaiii_config(
    *,
    pop_size: int,
    crossover: str,
    crossover_prob: float,
    crossover_eta: float,
    mutation: str,
    mutation_prob: str | float,
    mutation_eta: float,
    selection_pressure: int = 2,
    ref_divisions: int | None = None,
    engine: str,
    constraint_mode: str = "feasibility",
    track_genealogy: bool = False,
    result_mode: str = "non_dominated",
) -> dict[str, Any]:
    """Build NSGA-III config dict with all available parameters."""
    cfg = (
        NSGAIIIConfig()
        .pop_size(pop_size)
        .crossover(crossover, prob=crossover_prob, eta=crossover_eta)
        .mutation(mutation, prob=mutation_prob, eta=mutation_eta)
        .selection("tournament", pressure=selection_pressure)
        .engine(engine)
        .constraint_mode(constraint_mode)
        .track_genealogy(track_genealogy)
        .result_mode(result_mode)
    )
    if ref_divisions is not None:
        cfg = cfg.reference_directions(divisions=ref_divisions)
    return cfg.fixed().to_dict()

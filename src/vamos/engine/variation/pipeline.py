"""
VariationPipeline class leveraging shared registries/helpers.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.engine.variation.helpers import (
    validate_intensification_support,
    validate_operator_support,
)
from vamos.engine.variation.protocol import (
    CrossoverName,
    IntensificationName,
    IntensificationOperator,
    MutationName,
    RepairConfigValue,
    RepairName,
    RepairOperator,
    VariationWorkspaceProtocol,
)
from vamos.engine.variation.strategies import VariationContext, make_encoding_strategy
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.problem.types import ProblemProtocol


class VariationPipeline:
    """
    Encapsulates crossover + mutation (+ optional intensification + repair) for
    a given encoding.

    Existing pipelines without intensification preserve the legacy repair flow:
    crossover -> repair -> mutation -> repair. When intensification is enabled
    the stage order becomes crossover -> mutation -> intensification -> repair.
    """

    def __init__(
        self,
        *,
        encoding: EncodingLike,
        cross_method: CrossoverName,
        cross_params: dict[str, Any],
        mut_method: MutationName,
        mut_params: dict[str, Any],
        intensification_method: IntensificationName | None = None,
        intensification_params: dict[str, Any] | None = None,
        xl: np.ndarray,
        xu: np.ndarray,
        workspace: VariationWorkspaceProtocol | None,
        repair_cfg: RepairConfigValue = "auto",
        problem: ProblemProtocol | None = None,
    ) -> None:
        self.encoding = normalize_encoding(encoding)
        self.cross_method = cross_method
        self.cross_params = cross_params
        self.mut_method = mut_method
        self.mut_params = mut_params
        self.intensification_method = intensification_method
        self.intensification_params = intensification_params or {}
        self.xl = xl
        self.xu = xu
        self.workspace = workspace
        self.problem = problem
        self.repair_cfg = repair_cfg

        validate_operator_support(self.encoding, cross_method, mut_method)
        if intensification_method is not None:
            validate_intensification_support(self.encoding, intensification_method)

        ctx = VariationContext(xl=xl, xu=xu, workspace=workspace, problem=problem)
        strategy = make_encoding_strategy(self.encoding, ctx)
        self.parents_per_group = strategy.parents_per_group(cross_method)
        self.children_per_group = strategy.children_per_group(cross_method)

        # Build operators
        self.crossover_op = strategy.build_crossover(cross_method, cross_params)
        self.mutation_op = strategy.build_mutation(mut_method, mut_params)
        self.intensification_op = self._resolve_intensification(strategy)
        self.repair_op = self._resolve_repair()

    def _resolve_intensification(self, strategy: Any) -> IntensificationOperator | None:
        if self.intensification_method is None:
            return None
        return strategy.build_intensification(
            self.intensification_method,
            self.intensification_params,
            parents_per_group=self.parents_per_group,
            children_per_group=self.children_per_group,
        )

    def _resolve_repair(self) -> RepairOperator | None:
        if self.encoding != "real":
            if self.repair_cfg == "auto":
                return None
            raise ValueError("Repair operators are only supported for real encoding.")
        if self.repair_cfg == "auto":
            method: RepairName = "clip"
            params: dict[str, Any] = {}
        else:
            method, params = self.repair_cfg
        try:
            op_cls = cast(type[Any], get_operator_registry().get(method.lower()))
        except KeyError as exc:
            available = ", ".join(get_operator_registry().list())
            raise ValueError(f"Unknown repair operator '{method}'. Available: {available}") from exc
        try:
            op = op_cls(**params) if params else op_cls()
        except TypeError as exc:
            raise ValueError(f"Failed to initialize repair '{method}' with params {params}. Error: {exc}") from exc
        return cast(RepairOperator, op)

    def gather_parents(self, population: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
        if self.workspace is None:
            return cast(np.ndarray, population[parent_idx])
        shape = (parent_idx.size, population.shape[1])
        buffer = self.workspace.request("parent_buffer", shape, population.dtype)
        np.take(population, parent_idx, axis=0, out=buffer)
        return buffer

    def produce_offspring(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Crossover
        offspring = self.crossover_op(parents, rng)
        if self.intensification_op is None and self.encoding == "real":
            assert self.repair_op is not None
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)

        # Mutation
        offspring = self.mutation_op(offspring, rng)

        # Intensification
        if self.intensification_op is not None:
            offspring = self.intensification_op(offspring, rng, parents=parents)

        # Repair
        if self.repair_op is not None:
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)

        return offspring


__all__ = ["VariationPipeline"]

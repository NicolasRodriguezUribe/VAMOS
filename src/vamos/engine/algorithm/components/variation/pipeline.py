"""
VariationPipeline class leveraging shared registries/helpers.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vamos.operators.real import PolynomialMutation, SBXCrossover
from vamos.operators.real import (
    ArithmeticCrossover,
    PCXCrossover,
    SPXCrossover,
    UNDXCrossover,
    BLXAlphaCrossover,
    CauchyMutation,
    GaussianMutation,
    NonUniformMutation,
    LinkedPolynomialMutation,
    UniformMutation,
    UniformResetMutation,
    VariationWorkspace,
    ClampRepair,
    ReflectRepair,
    ResampleRepair,
    RoundRepair,
)
from vamos.engine.algorithm.components.variation.helpers import (
    PERM_CROSSOVER,
    PERM_MUTATION,
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
    validate_operator_support,
)
from vamos.operators.integer import creep_mutation


class VariationPipeline:
    """
    Encapsulates crossover + mutation (+ optional repair) for a given encoding.
    """

    def __init__(
        self,
        *,
        encoding: str,
        cross_method: str,
        cross_params: dict[str, Any],
        mut_method: str,
        mut_params: dict[str, Any],
        xl: np.ndarray,
        xu: np.ndarray,
        workspace: VariationWorkspace | None,
        repair_cfg: tuple[str, dict[str, Any]] | None = None,
        problem: Any | None = None,
    ) -> None:
        self.encoding = encoding
        self.cross_method = cross_method
        self.cross_params = cross_params
        self.mut_method = mut_method
        self.mut_params = mut_params
        self.xl = xl
        self.xu = xu
        self.workspace = workspace
        self.problem = problem
        self.repair_cfg = repair_cfg
        self.parents_per_group = self._parents_required(cross_method)
        self.children_per_group = self.parents_per_group
        validate_operator_support(encoding, cross_method, mut_method)
        self.crossover_op: Any = self._build_crossover_operator()
        self.mutation_op: Any = self._build_mutation_operator()
        self.repair_op: Any = self._build_repair_operator()

    def gather_parents(self, population: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
        if self.workspace is None:
            return cast(np.ndarray, population[parent_idx])
        shape = (parent_idx.size, population.shape[1])
        buffer = self.workspace.request("parent_buffer", shape, population.dtype)
        np.take(population, parent_idx, axis=0, out=buffer)
        return buffer

    def produce_offspring(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        encoding = self.encoding
        cross_params = self.cross_params
        mut_params = self.mut_params
        xl, xu = self.xl, self.xu
        group_size = self.parents_per_group
        if encoding == "permutation":
            cross_prob = float(cross_params.get("prob", 1.0))
            cross_fn = cast(Any, PERM_CROSSOVER[self.cross_method])
            offspring = cast(np.ndarray, cross_fn(parents, cross_prob, rng))
            mut_fn = cast(Any, PERM_MUTATION[self.mut_method])
            mut_fn(offspring, float(mut_params.get("prob", 0.0)), rng)
            return offspring

        if encoding == "binary":
            cross_prob = float(cross_params.get("prob", 1.0))
            cross_fn = cast(Any, BINARY_CROSSOVER[self.cross_method])
            offspring = cast(np.ndarray, cross_fn(parents, cross_prob, rng))
            mut_fn = cast(Any, BINARY_MUTATION[self.mut_method])
            mut_fn(offspring, float(mut_params.get("prob", 0.0)), rng)
            return offspring

        if encoding == "integer":
            cross_prob = float(cross_params.get("prob", 1.0))
            cross_fn = cast(Any, INT_CROSSOVER[self.cross_method])
            offspring = cast(np.ndarray, cross_fn(parents, cross_prob, rng))
            mut_fn = cast(Any, INT_MUTATION[self.mut_method])
            mut_prob = float(mut_params.get("prob", 0.0))
            if mut_fn is creep_mutation:
                step = int(mut_params.get("step", 1))
                mut_fn(offspring, mut_prob, step, xl, xu, rng)
            else:
                mut_fn(offspring, mut_prob, xl, xu, rng)
            return offspring

        if encoding == "mixed":
            spec = getattr(self.problem, "mixed_spec", None)
            if spec is None:
                raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
            cross_prob = float(cross_params.get("prob", 1.0))
            cross_fn = cast(Any, MIXED_CROSSOVER[self.cross_method])
            offspring = cast(np.ndarray, cross_fn(parents, cross_prob, spec, rng))
            mut_fn = cast(Any, MIXED_MUTATION[self.mut_method])
            mut_fn(offspring, float(mut_params.get("prob", 0.0)), spec, rng)
            return offspring

        n_var = parents.shape[1]
        if self.crossover_op is None:
            raise ValueError("Crossover operator is not initialized.")
        if parents.shape[0] % group_size != 0:
            usable = (parents.shape[0] // group_size) * group_size
            parents = parents[:usable]
        if parents.size == 0:
            return parents
        if group_size == 2:
            pairs = parents.reshape(-1, 2, n_var)
            offspring = cast(np.ndarray, self.crossover_op(pairs, rng)).reshape(parents.shape)
        else:
            groups = parents.reshape(-1, group_size, n_var)
            offspring = cast(np.ndarray, self.crossover_op(groups, rng)).reshape(parents.shape)

        if self.mutation_op is None:
            raise ValueError("Mutation operator is not initialized.")
        offspring = cast(np.ndarray, self.mutation_op(offspring, rng))

        if self.repair_op is not None:
            flat = offspring.reshape(-1, offspring.shape[-1])
            repaired = cast(np.ndarray, self.repair_op(flat, xl, xu, rng))
            offspring = repaired.reshape(offspring.shape)
        return offspring

    def _build_crossover_operator(self) -> Any | None:
        encoding = self.encoding
        method = self.cross_method
        params = self.cross_params
        xl, xu = self.xl, self.xu
        workspace = self.workspace
        if encoding in {"permutation", "binary", "integer", "mixed"}:
            return None
        if method == "sbx":
            prob = float(params.get("prob", 0.9))
            eta = float(params.get("eta", 20.0))
            return SBXCrossover(
                prob_crossover=prob,
                eta=eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
                allow_inplace=True,
            )
        if method == "blx_alpha":
            prob = float(params.get("prob", 1.0))
            alpha = float(params.get("alpha", 0.5))
            repair = params.get("repair", "clip")
            return BLXAlphaCrossover(
                alpha=alpha,
                prob_crossover=prob,
                lower=xl,
                upper=xu,
                repair=repair,
                workspace=workspace,
                allow_inplace=True,
            )
        if method == "arithmetic":
            prob = float(params.get("prob", 0.9))
            return ArithmeticCrossover(prob_crossover=prob)
        if method == "pcx":
            sigma_eta = float(params.get("sigma_eta", 0.1))
            sigma_zeta = float(params.get("sigma_zeta", 0.1))
            return PCXCrossover(
                sigma_eta=sigma_eta,
                sigma_zeta=sigma_zeta,
                lower=xl,
                upper=xu,
            )
        if method == "undx":
            sigma_xi = float(params.get("sigma_xi", 0.5))
            sigma_eta = float(params.get("sigma_eta", 0.35))
            return UNDXCrossover(
                sigma_xi=sigma_xi,
                sigma_eta=sigma_eta,
                lower=xl,
                upper=xu,
            )
        if method == "spx":
            epsilon = float(params.get("epsilon", 0.5))
            return SPXCrossover(
                epsilon=epsilon,
                lower=xl,
                upper=xu,
            )
        return None

    def _build_mutation_operator(self) -> Any | None:
        encoding = self.encoding
        method = self.mut_method
        params = self.mut_params
        xl, xu = self.xl, self.xu
        workspace = self.workspace
        n_var = xl.shape[0]
        if encoding in {"permutation", "binary", "integer", "mixed"}:
            return None
        if method == "pm":
            prob = float(params.get("prob", 0.1))
            eta = float(params.get("eta", 20.0))
            return PolynomialMutation(
                prob_mutation=prob,
                eta=eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )
        if method == "non_uniform":
            prob = float(params.get("prob", 0.1))
            perturb = float(params.get("perturbation", 0.5))
            return NonUniformMutation(
                prob_mutation=prob,
                perturbation=perturb,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )
        if method == "gaussian":
            prob = float(params.get("prob", 0.1))
            sigma = float(params.get("sigma", 0.1))
            return GaussianMutation(prob_mutation=prob, sigma=sigma, lower=xl, upper=xu)
        if method == "uniform_reset":
            prob = float(params.get("prob", 0.1))
            return UniformResetMutation(prob_mutation=prob, lower=xl, upper=xu)
        if method == "cauchy":
            prob = float(params.get("prob", 0.1))
            gamma = float(params.get("gamma", 0.1))
            return CauchyMutation(prob_mutation=prob, gamma=gamma, lower=xl, upper=xu)
        if method == "uniform":
            prob = float(params.get("prob", 1.0 / max(1, n_var)))
            perturb = float(params.get("perturb", 0.1))
            return UniformMutation(prob=prob, perturb=perturb, lower=xl, upper=xu, repair=None)
        if method == "linked_polynomial":
            prob = float(params.get("prob", 1.0 / max(1, n_var)))
            eta = float(params.get("eta", 20.0))
            return LinkedPolynomialMutation(prob=prob, eta=eta, lower=xl, upper=xu, repair=None)
        return None

    def _build_repair_operator(self) -> Any | None:
        encoding = self.encoding
        if encoding in {"permutation", "binary", "integer", "mixed"}:
            return None
        repair_cfg = self.repair_cfg
        if not repair_cfg:
            return None
        method, _params = repair_cfg
        normalized = method.lower()
        if normalized in {"clip", "clamp"}:
            return ClampRepair()
        if normalized == "reflect":
            return ReflectRepair()
        if normalized in {"random", "resample"}:
            return ResampleRepair()
        if normalized == "round":
            return RoundRepair()
        raise ValueError(f"Unsupported repair strategy '{method}'.")

    @staticmethod
    def _parents_required(cross_method: str) -> int:
        if cross_method in {"pcx", "undx", "spx"}:
            return 3
        return 2


__all__ = ["VariationPipeline"]

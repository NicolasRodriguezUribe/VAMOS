"""
VariationPipeline class leveraging shared registries/helpers.
"""

from __future__ import annotations

from typing import Any, cast, Callable

import numpy as np

from vamos.operators.registry import operator_registry
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

# Protocol definition location
from vamos.engine.algorithm.components.variation.protocol import VariationOperator

class VariationPipeline:
    """
    Encapsulates crossover + mutation (+ optional repair) for a given encoding.
    Now leverages `operator_registry` for dynamic resolution of real-valued operators.
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
        workspace: Any | None,
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
        
        # Determine parents required (legacy helper needed?)
        # Ideally operators should report this.
        self.parents_per_group = self._parents_required(cross_method)
        self.children_per_group = self.parents_per_group
        
        validate_operator_support(encoding, cross_method, mut_method)
        
        # Build operators
        self.crossover_op: Any = self._resolve_operator(cross_method, cross_params, is_crossover=True)
        self.mutation_op: Any = self._resolve_operator(mut_method, mut_params, is_crossover=False)
        self.repair_op: Any = self._resolve_repair()

    def _resolve_operator(self, method: str, params: dict[str, Any], is_crossover: bool) -> Any | None:
        """
        Resolve operator from registry if applicable (real encoding).
        Returns None for specialized encodings handled in produce_offspring (legacy).
        """
        if self.encoding in {"permutation", "binary", "integer", "mixed"}:
            return None
            
        try:
            op_cls = operator_registry.get(method)
        except KeyError:
            # Fallback or error?
            return None

        # Build kwargs for operator constructor
        # We need to map generic params to specific constructor args.
        # This mapping logic was previously hardcoded.
        # Ideally, we standardize params.
        
        # Common args
        kwargs = {
            "lower": self.xl,
            "upper": self.xu,
            "workspace": self.workspace,
        }
        
        # Specific mappings (Compatibility handling)
        if is_crossover:
            kwargs["prob_crossover"] = float(params.get("prob", 0.9))
            kwargs["allow_inplace"] = True
            # Mapping specific params
            if method == "sbx":
                kwargs["eta"] = float(params.get("eta", 20.0))
            if method == "blx_alpha":
                 kwargs["alpha"] = float(params.get("alpha", 0.5))
                 kwargs["repair"] = params.get("repair", "clip")
            if method == "pcx":
                kwargs["sigma_eta"] = float(params.get("sigma_eta", 0.1))
                kwargs["sigma_zeta"] = float(params.get("sigma_zeta", 0.1))
            if method == "undx":
                kwargs["sigma_xi"] = float(params.get("sigma_xi", 0.5))
                kwargs["sigma_eta"] = float(params.get("sigma_eta", 0.35))
            if method == "spx":
                kwargs["epsilon"] = float(params.get("epsilon", 0.5))
        else:
             kwargs["prob_mutation"] = float(params.get("prob", 0.1))
             if method in {"pm", "polynomial", "linked_polynomial"}:
                 kwargs["eta"] = float(params.get("eta", 20.0))
             if method == "non_uniform":
                 kwargs["perturbation"] = float(params.get("perturbation", 0.5))
             if method == "gaussian":
                 kwargs["sigma"] = float(params.get("sigma", 0.1))
             if method == "cauchy":
                 kwargs["gamma"] = float(params.get("gamma", 0.1))
             if method == "uniform":
                 kwargs["perturb"] = float(params.get("perturb", 0.1))
                 kwargs["repair"] = None

        # Filter kwargs that the constructor accepts?
        # Python classes will error on unexpected kwargs unless they take **kwargs.
        # Most VAMOS operators are specific.
        # We need to be careful here or accept that we are cleaning up.
        # Let's simple check if method is in registry and instantiate.
        
        # HACK: Filter kwargs based on known keys for safety or just pass relevant ones?
        # Relying on implicit knowledge of operator constructor is what makes this hard to genericize fully without a Factory pattern.
        # But `operator_registry` contains classes.
        
        return op_cls(**{k: v for k, v in kwargs.items() if k in op_cls.__init__.__code__.co_varnames})

    def _resolve_repair(self) -> Any | None:
        if self.encoding in {"permutation", "binary", "integer", "mixed"}:
            return None
        if not self.repair_cfg:
            return None
        method, _ = self.repair_cfg
        try:
            return operator_registry.get(method.lower())()
        except KeyError:
            return None

    def gather_parents(self, population: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
        if self.workspace is None:
            return cast(np.ndarray, population[parent_idx])
        shape = (parent_idx.size, population.shape[1])
        buffer = self.workspace.request("parent_buffer", shape, population.dtype)
        np.take(population, parent_idx, axis=0, out=buffer)
        return buffer

    def produce_offspring(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Legacy handling for non-real encodings (unchanged for now)
        if self.encoding != "real":
             return self._produce_offspring_legacy(parents, rng)

        # Pipeline execution for Real encoding
        n_var = parents.shape[1]
        group_size = self.parents_per_group
        
        # Crossover
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

        # Mutation
        if self.mutation_op is None:
            raise ValueError("Mutation operator is not initialized.")
        offspring = cast(np.ndarray, self.mutation_op(offspring, rng))

        # Repair
        if self.repair_op is not None:
            flat = offspring.reshape(-1, offspring.shape[-1])
            repaired = cast(np.ndarray, self.repair_op(flat, self.xl, self.xu, rng))
            offspring = repaired.reshape(offspring.shape)
            
        return offspring

    def _produce_offspring_legacy(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Copied legacy logic for other encodings
        encoding = self.encoding
        cross_params = self.cross_params
        mut_params = self.mut_params
        xl, xu = self.xl, self.xu
        
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
            
        return parents

    @staticmethod
    def _parents_required(cross_method: str) -> int:
        if cross_method in {"pcx", "undx", "spx"}:
            return 3
        return 2

__all__ = ["VariationPipeline"]

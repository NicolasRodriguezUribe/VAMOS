# algorithm/nsgaii.py
import numpy as np

from vamos.algorithm.archive import CrowdingDistanceArchive
from vamos.operators.permutation import (
    order_crossover,
    random_permutation_population,
    swap_mutation,
)
from vamos.operators.real import (
    SBXCrossover,
    PolynomialMutation,
    BLXAlphaCrossover,
    NonUniformMutation,
    VariationWorkspace,
    ClampRepair,
    ReflectRepair,
    ResampleRepair,
    RoundRepair,
)
from vamos.algorithm.hypervolume import hypervolume


def _resolve_prob_expression(value, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


def _initialize_population(
    pop_size: int,
    n_var: int,
    xl,
    xu,
    encoding: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if pop_size <= 0:
        raise ValueError("pop_size must be positive.")
    if encoding == "permutation":
        return random_permutation_population(pop_size, n_var, rng)
    return rng.uniform(xl, xu, size=(pop_size, n_var))


def _evaluate_population(problem, X: np.ndarray) -> np.ndarray:
    F = np.empty((X.shape[0], problem.n_obj))
    problem.evaluate(X, {"F": F})
    return F


def _prepare_mutation_params(mut_params: dict, encoding: str, n_var: int) -> dict:
    params = dict(mut_params)
    if "prob" in params:
        params["prob"] = _resolve_prob_expression(params["prob"], n_var, params["prob"])
    else:
        if encoding == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
    return params


def _build_mating_pool(
    kernel,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    parent_count: int,
) -> np.ndarray:
    if parent_count % 2 != 0:
        raise ValueError("NSGA-II requires an even offspring size.")
    parent_indices = kernel.tournament_selection(
        ranks, crowding, pressure, rng, n_parents=parent_count
    )
    if parent_indices.size != parent_count:
        raise ValueError("Selection operator returned an unexpected number of parents.")
    return parent_indices.reshape(parent_count // 2, 2)


class NSGAII:
    """
    Vectorized/SOA-style NSGA-II evolutionary core.
    Individuals are represented as array rows (X, F) without per-object instances.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int):
        term_type, term_val = termination
        hv_target = None
        hv_ref_point = None
        hv_evaluator = None
        if term_type == "n_eval":
            max_eval = int(term_val)
        elif term_type == "hv":
            hv_target = float(term_val["target_value"])
            hv_ref_point = np.asarray(term_val["reference_point"], dtype=float)
            max_eval = int(term_val.get("max_evaluations", 0))
            if max_eval <= 0:
                raise ValueError("HV-based termination requires a positive max_evaluations value.")
            if self.kernel.supports_quality_indicator("hypervolume"):
                hv_evaluator = self.kernel.hypervolume
            else:
                hv_evaluator = hypervolume
        else:
            raise ValueError("Unsupported termination criterion for NSGA-II.")
        if hv_target is not None and hv_evaluator is None:
            hv_evaluator = hypervolume

        rng = np.random.default_rng(seed)
        pop_size = int(self.cfg["pop_size"])
        offspring_size = int(self.cfg.get("offspring_size") or pop_size)
        if offspring_size <= 0:
            raise ValueError("offspring size must be positive.")
        if offspring_size % 2 != 0:
            raise ValueError("offspring size must be even.")
        encoding = getattr(problem, "encoding", "continuous")
        n_var = problem.n_var
        xl = np.asarray(problem.xl, dtype=float)
        xu = np.asarray(problem.xu, dtype=float)
        if xl.ndim == 0:
            xl = np.full(n_var, xl, dtype=float)
        if xu.ndim == 0:
            xu = np.full(n_var, xu, dtype=float)
        xl = np.ascontiguousarray(xl, dtype=float)
        xu = np.ascontiguousarray(xu, dtype=float)

        X = _initialize_population(pop_size, n_var, xl, xu, encoding, rng)
        F = _evaluate_population(problem, X)
        n_eval = X.shape[0]
        archive_size = self._resolve_archive_size()
        archive_X = archive_F = None
        archive_manager = None
        archive_via_kernel = False
        if archive_size:
            if hasattr(self.kernel, "update_archive"):
                archive_via_kernel = True
                archive_X, archive_F = self.kernel.update_archive(
                    None, None, X, F, archive_size
                )
            else:
                archive_manager = CrowdingDistanceArchive(
                    archive_size, n_var, problem.n_obj, X.dtype
                )
                archive_X, archive_F = archive_manager.update(X, F)

        sel_method, sel_params = self.cfg["selection"]
        if sel_method != "tournament":
            raise ValueError(f"Unsupported selection method '{sel_method}'.")
        pressure = int(sel_params.get("pressure", 2))

        cross_method, cross_params = self.cfg["crossover"]
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = _prepare_mutation_params(mut_params, encoding, n_var)

        variation_workspace = VariationWorkspace() if encoding != "permutation" else None

        self._validate_operator_support(encoding, cross_method, mut_method)
        crossover_operator = self._build_crossover_operator(
            encoding, cross_method, cross_params, xl, xu, variation_workspace
        )
        mutation_operator = self._build_mutation_operator(
            encoding, mut_method, mut_params, xl, xu, variation_workspace
        )
        repair_cfg = self.cfg.get("repair")
        repair_operator = self._build_repair_operator(encoding, repair_cfg)
        hv_reached = False
        if hv_target is not None:
            current_hv = hv_evaluator(F, hv_ref_point)
            if current_hv >= hv_target:
                hv_reached = True

        while n_eval < max_eval and not hv_reached:
            ranks, crowding = self.kernel.nsga2_ranking(F)
            mating_pairs = _build_mating_pool(
                self.kernel, ranks, crowding, pressure, rng, offspring_size
            )
            parent_idx = mating_pairs.reshape(-1)
            X_parents = self._gather_parents(X, parent_idx, variation_workspace)

            X_off = self._produce_offspring(
                X_parents,
                encoding,
                cross_method,
                cross_params,
                mut_method,
                mut_params,
                rng,
                xl,
                xu,
                crossover_operator,
                mutation_operator,
                repair_operator,
            )

            F_off = _evaluate_population(problem, X_off)
            n_eval += X_off.shape[0]

            X, F = self.kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
            if archive_size:
                if archive_via_kernel:
                    archive_X, archive_F = self.kernel.update_archive(
                        archive_X, archive_F, X, F, archive_size
                    )
                elif archive_manager is not None:
                    archive_X, archive_F = archive_manager.update(X, F)
            if hv_target is not None:
                current_hv = hv_evaluator(F, hv_ref_point)
                if current_hv >= hv_target:
                    hv_reached = True
                    break

        result = {"X": X, "F": F, "evaluations": n_eval, "hv_reached": hv_reached}
        if archive_size:
            if archive_manager is not None:
                final_X, final_F = archive_manager.contents()
                result["archive"] = {"X": final_X, "F": final_F}
            elif archive_via_kernel:
                result["archive"] = {"X": archive_X, "F": archive_F}
        return result

    @staticmethod
    def _validate_operator_support(encoding: str, crossover: str, mutation: str) -> None:
        if encoding == "permutation":
            if crossover not in {"ox", "order"}:
                raise ValueError(f"Unsupported crossover '{crossover}' for permutation encoding.")
            if mutation != "swap":
                raise ValueError(f"Unsupported mutation '{mutation}' for permutation encoding.")
        else:
            if crossover not in {"sbx", "blx_alpha"}:
                raise ValueError(f"Unsupported crossover '{crossover}' for continuous encoding.")
            if mutation not in {"pm", "non_uniform"}:
                raise ValueError(f"Unsupported mutation '{mutation}' for continuous encoding.")

    def _gather_parents(
        self,
        population: np.ndarray,
        parent_idx: np.ndarray,
        workspace: VariationWorkspace | None,
    ) -> np.ndarray:
        if workspace is None:
            return population[parent_idx]
        shape = (parent_idx.size, population.shape[1])
        buffer = workspace.request("parent_buffer", shape, population.dtype)
        np.take(population, parent_idx, axis=0, out=buffer)
        return buffer

    def _produce_offspring(
        self,
        parents: np.ndarray,
        encoding: str,
        crossover: str,
        cross_params: dict,
        mutation: str,
        mut_params: dict,
        rng: np.random.Generator,
        xl,
        xu,
        crossover_operator,
        mutation_operator,
        repair_operator,
    ) -> np.ndarray:
        if encoding == "permutation":
            cross_prob = float(cross_params.get("prob", 1.0))
            offspring = order_crossover(parents, cross_prob, rng)
            swap_mutation(offspring, float(mut_params.get("prob", 0.0)), rng)
            return offspring

        n_var = parents.shape[1]
        if crossover_operator is None:
            raise ValueError("Crossover operator is not initialized.")
        pairs = parents.reshape(-1, 2, n_var)
        offspring = crossover_operator(pairs, rng).reshape(parents.shape)

        if mutation_operator is None:
            raise ValueError("Mutation operator is not initialized.")
        offspring = mutation_operator(offspring, rng)

        if repair_operator is not None and encoding != "permutation":
            flat = offspring.reshape(-1, offspring.shape[-1])
            repaired = repair_operator(flat, xl, xu, rng)
            offspring = repaired.reshape(offspring.shape)
        return offspring

    def _build_crossover_operator(self, encoding, method, params, xl, xu, workspace):
        if encoding == "permutation":
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
        return None

    def _build_mutation_operator(self, encoding, method, params, xl, xu, workspace):
        if encoding == "permutation":
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
        return None

    def _build_repair_operator(self, encoding, repair_cfg):
        if encoding == "permutation" or not repair_cfg:
            return None
        method, _ = repair_cfg
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

    def _resolve_archive_size(self) -> int | None:
        archive_cfg = self.cfg.get("archive")
        if not archive_cfg:
            return None
        size = int(archive_cfg.get("size", 0))
        return size if size > 0 else None

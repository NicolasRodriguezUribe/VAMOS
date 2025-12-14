from __future__ import annotations

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning.core.parameter_space import (
    AlgorithmConfigSpace,
    Categorical,
    CategoricalInteger,
    Double,
    Integer,
    ParameterDefinition,
)


def build_nsgaii_config_space(
    *,
    crossover_methods=("sbx", "blx_alpha", "arithmetic", "pcx", "undx", "spx"),
    mutation_methods=("pm", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform", "linked_polynomial"),
    pop_size_values=(40, 80, 120),
    eta_choices=(10.0, 20.0, 30.0),
    crossover_prob_range=(0.7, 0.95),
    mutation_prob_expr="1/n",
    blx_alpha_range=(0.1, 0.9),
    non_uniform_perturb_range=(0.1, 1.0),
    gaussian_sigma_range=(0.01, 0.5),
    pcx_sigma_eta_range=(0.05, 0.2),
    pcx_sigma_zeta_range=(0.05, 0.2),
    undx_sigma_xi_range=(0.1, 0.7),
    undx_sigma_eta_range=(0.1, 0.7),
    spx_epsilon_range=(0.1, 1.0),
    cauchy_gamma_range=(0.01, 0.5),
    archive_size_range=(0, 300),
    mutation_prob_factor_range=(0.2, 2.0),
    initializer_choices=("random", "lhs", "scatter_search"),
    result_modes=("population", "external_archive"),
    selection_methods=("tournament", "random"),
    selection_pressure_values=(2, 3, 4),
    repair_methods=("clip", "reflect", "resample", "round"),
) -> AlgorithmConfigSpace:
    """
    Convenience builder for an NSGA-II hyperparameter search space using the
    generic AlgorithmConfigSpace/ParameterDefinition API.
    """
    if not pop_size_values:
        raise ValueError("pop_size_values must contain at least one entry.")
    pop_values = tuple(int(v) for v in pop_size_values)

    cross_methods = tuple(dict.fromkeys(str(m).lower() for m in crossover_methods))
    if not cross_methods:
        raise ValueError("At least one crossover method must be provided.")
    mut_methods = tuple(dict.fromkeys(str(m).lower() for m in mutation_methods))
    if not mut_methods:
        raise ValueError("At least one mutation method must be provided.")

    eta_vals = tuple(float(e) for e in eta_choices)
    if not eta_vals:
        raise ValueError("eta_choices must contain at least one value.")
    eta_min, eta_max = min(eta_vals), max(eta_vals)

    cross_prob_low, cross_prob_high = map(float, crossover_prob_range)
    if cross_prob_low > cross_prob_high:
        raise ValueError("crossover_prob_range must be ordered (low, high).")

    mut_prob_options = ["1/n"]
    if mutation_prob_expr not in mut_prob_options:
        mut_prob_options.append(str(mutation_prob_expr))
    archive_min, archive_max = map(int, archive_size_range)
    if archive_min < 0 or archive_max < 0:
        raise ValueError("archive_size_range must be non-negative.")
    if archive_min > archive_max:
        raise ValueError("archive_size_range must be ordered (low, high).")

    crossover_branch: dict[str, dict[str, ParameterDefinition]] = {}
    if "sbx" in cross_methods:
        crossover_branch["sbx"] = {"eta": ParameterDefinition(Double(eta_min, eta_max))}
    if "blx_alpha" in cross_methods:
        blx_alpha_low, blx_alpha_high = map(float, blx_alpha_range)
        if blx_alpha_low > blx_alpha_high:
            raise ValueError("blx_alpha_range must be ordered (low, high).")
        crossover_branch["blx_alpha"] = {
            "alpha": ParameterDefinition(Double(blx_alpha_low, blx_alpha_high)),
            "repair": ParameterDefinition(Categorical(tuple(dict.fromkeys(repair_methods)))),
        }
    if "arithmetic" in cross_methods:
        crossover_branch["arithmetic"] = {}
    if "pcx" in cross_methods:
        eta_low, eta_high = map(float, pcx_sigma_eta_range)
        zeta_low, zeta_high = map(float, pcx_sigma_zeta_range)
        if eta_low > eta_high or zeta_low > zeta_high:
            raise ValueError("pcx sigma ranges must be ordered (low, high).")
        crossover_branch["pcx"] = {
            "sigma_eta": ParameterDefinition(Double(eta_low, eta_high)),
            "sigma_zeta": ParameterDefinition(Double(zeta_low, zeta_high)),
        }
    if "undx" in cross_methods:
        xi_low, xi_high = map(float, undx_sigma_xi_range)
        eta_low, eta_high = map(float, undx_sigma_eta_range)
        if xi_low > xi_high or eta_low > eta_high:
            raise ValueError("undx sigma ranges must be ordered (low, high).")
        crossover_branch["undx"] = {
            "sigma_xi": ParameterDefinition(Double(xi_low, xi_high)),
            "sigma_eta": ParameterDefinition(Double(eta_low, eta_high)),
        }
    if "spx" in cross_methods:
        eps_low, eps_high = map(float, spx_epsilon_range)
        if eps_low > eps_high:
            raise ValueError("spx_epsilon_range must be ordered (low, high).")
        crossover_branch["spx"] = {"epsilon": ParameterDefinition(Double(eps_low, eps_high))}

    mutation_branch: dict[str, dict[str, ParameterDefinition]] = {}
    if "pm" in mut_methods:
        mutation_branch["pm"] = {"eta": ParameterDefinition(Double(eta_min, eta_max))}
    if "non_uniform" in mut_methods:
        perturb_low, perturb_high = map(float, non_uniform_perturb_range)
        if perturb_low > perturb_high:
            raise ValueError("non_uniform_perturb_range must be ordered (low, high).")
        mutation_branch["non_uniform"] = {
            "perturbation": ParameterDefinition(Double(perturb_low, perturb_high))
        }
    if "gaussian" in mut_methods:
        sigma_low, sigma_high = map(float, gaussian_sigma_range)
        if sigma_low <= 0.0 or sigma_high <= 0.0:
            raise ValueError("gaussian_sigma_range values must be positive.")
        if sigma_low > sigma_high:
            raise ValueError("gaussian_sigma_range must be ordered (low, high).")
        mutation_branch["gaussian"] = {"sigma": ParameterDefinition(Double(sigma_low, sigma_high))}
    if "uniform_reset" in mut_methods:
        mutation_branch["uniform_reset"] = {}
    if "cauchy" in mut_methods:
        gamma_low, gamma_high = map(float, cauchy_gamma_range)
        if gamma_low <= 0.0 or gamma_high <= 0.0:
            raise ValueError("cauchy_gamma_range values must be positive.")
        if gamma_low > gamma_high:
            raise ValueError("cauchy_gamma_range must be ordered (low, high).")
        mutation_branch["cauchy"] = {"gamma": ParameterDefinition(Double(gamma_low, gamma_high))}
    if "uniform" in mut_methods:
        mutation_branch["uniform"] = {"perturb": ParameterDefinition(Double(0.01, 1.0))}
    if "linked_polynomial" in mut_methods:
        mutation_branch["linked_polynomial"] = {"eta": ParameterDefinition(Double(eta_min, eta_max))}

    params = {
        "pop_size": ParameterDefinition(CategoricalInteger(pop_values)),
        "crossover": ParameterDefinition(
            Categorical(cross_methods),
            sub_parameters={"prob": ParameterDefinition(Double(cross_prob_low, cross_prob_high))},
            conditional_sub_parameters=crossover_branch,
        ),
        "mutation": ParameterDefinition(
            Categorical(mut_methods),
            sub_parameters={"prob": ParameterDefinition(Categorical(mut_prob_options))},
            conditional_sub_parameters=mutation_branch,
        ),
        "mutation_prob_factor": ParameterDefinition(
            Double(float(mutation_prob_factor_range[0]), float(mutation_prob_factor_range[1])),
            fixed_sub_parameters={},
        ),
        "selection": ParameterDefinition(
            Categorical(tuple(selection_methods)),
            sub_parameters={},
            conditional_sub_parameters={
                "tournament": {
                    "pressure": ParameterDefinition(
                        CategoricalInteger(tuple(int(p) for p in selection_pressure_values))
                    )
                }
            },
        ),
        "archive": ParameterDefinition(Integer(archive_min, archive_max)),
        "initializer": ParameterDefinition(Categorical(tuple(initializer_choices))),
        "result_mode": ParameterDefinition(Categorical(tuple(result_modes))),
    }

    fixed = {"survival": "nsga2", "engine": "numpy"}
    return AlgorithmConfigSpace(NSGAIIConfig, params, fixed_values=fixed)


__all__ = ["build_nsgaii_config_space"]

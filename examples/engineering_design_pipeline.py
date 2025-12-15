"""
Mixed-encoding engineering design example (welded beam surrogate).

Usage:
    python examples/engineering_design_pipeline.py

Requirements:
    pip install -e ".[examples]"  # matplotlib
"""
from __future__ import annotations

from vamos import (
    NSGAIIConfig,
    OptimizeConfig,
    WeldedBeamDesignProblem,
    optimize,
    plot_pareto_front_2d,
)


def build_config(pop_size: int = 30) -> dict:
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .constraint_mode("feasibility")
        .fixed()
    )
    return cfg.to_dict()


def main(seed: int = 11) -> None:
    problem = WeldedBeamDesignProblem()
    cfg = build_config(pop_size=24)
    result = optimize(
        OptimizeConfig(
            problem=problem,
            algorithm="nsgaii",
            algorithm_config=cfg,
            termination=("n_eval", 200),
            seed=seed,
            engine="numpy",
        )
    )
    F = result.F
    G = result.data.get("G")
    print(f"Collected {F.shape[0]} design candidates.")

    feasible_mask = None
    if G is not None:
        feasible_mask = (G <= 0).all(axis=1)
        print(f"Feasible designs: {int(feasible_mask.sum())}/{F.shape[0]}")

    try:
        import matplotlib.pyplot as plt

        to_plot = F if feasible_mask is None else F[feasible_mask]
        plot_pareto_front_2d(
            to_plot,
            labels=("Fabrication cost", "Deflection proxy"),
            title="Welded beam design trade-offs",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting optional in CI
        print("Plotting skipped:", exc)


if __name__ == "__main__":
    main()

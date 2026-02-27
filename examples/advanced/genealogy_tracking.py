"""
Genealogy tracking with NSGA-II.

When track_genealogy is enabled, VAMOS records every parent-child relationship
throughout the optimization run. Each individual gets a unique ID, and the
tracker stores which parents produced it, which genetic operator was used
(e.g. "sbx+pm"), and in which generation it was created.

At the end of the run, VAMOS traces back the full ancestry of every solution
in the final Pareto front and computes two analytics:

  - operator_stats: For each genetic operator, how many times it was used in
    total vs. how many of those uses appear in the lineage of final front
    solutions. A higher ratio means the operator was more "effective" at
    producing ancestors of good solutions.

  - generation_contributions: For each generation, how many individuals were
    created in total vs. how many appear in the lineage of final front
    solutions. This shows which generations contributed the most genetic
    material to the final result.

Usage:
    python examples/advanced/genealogy_tracking.py
"""

from __future__ import annotations

from vamos import optimize
from vamos.algorithms import NSGAIIConfig


def main() -> None:
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .initializer("sobol", scramble=True)
        # Enable genealogy: records parent IDs, operator name, and generation
        # for every individual created during the run.
        .track_genealogy(True)
        .build()
    )

    print("Running NSGA-II with genealogy tracking on DTLZ2 ...")
    result = optimize(
        "dtlz2",
        algorithm="nsgaii",
        algorithm_config=config,
        max_evaluations=30_000,
        seed=42,
        engine="numba",
    )

    F = result.F
    print(f"Found {len(F)} Pareto-optimal solutions")

    # Genealogy data is stored under result.data["genealogy"] when enabled.
    genealogy = result.data.get("genealogy")
    if genealogy is None:
        print("No genealogy data (should not happen with track_genealogy=True)")
        return

    # operator_stats: list of dicts, one per operator variant.
    #   - "operator": name string (e.g. "sbx+pm")
    #   - "total_uses": how many offspring this operator produced across all gens
    #   - "uses_in_final_lineages": how many of those are ancestors of the final front
    #   - "ratio": uses_in_final_lineages / total_uses (higher = more effective)
    operator_stats = genealogy["operator_stats"]

    # generation_contributions: list of dicts, one per generation.
    #   - "generation": generation number (0 = initial population)
    #   - "total": individuals created in this generation
    #   - "final_lineage": how many are ancestors of the final Pareto front
    #   - "ratio": final_lineage / total (higher = this generation contributed more)
    gen_contributions = genealogy["generation_contributions"]

    print("\n-- Operator effectiveness --")
    for row in operator_stats:
        print(
            f"  {row['operator']:>20s}  |  "
            f"total={row['total_uses']:>5}  |  "
            f"in final lineages={row['uses_in_final_lineages']:>5}  |  "
            f"ratio={row['ratio']:.3f}"
        )

    print(f"\n-- Generation contributions ({len(gen_contributions)} generations) --")
    for row in gen_contributions[:5]:
        print(
            f"  gen {row['generation']:>3}  |  "
            f"total={row['total']:>4}  |  "
            f"final lineage={row['final_lineage']:>4}  |  "
            f"ratio={row['ratio']:.3f}"
        )
    if len(gen_contributions) > 10:
        print("  ...")
    for row in gen_contributions[-3:]:
        print(
            f"  gen {row['generation']:>3}  |  "
            f"total={row['total']:>4}  |  "
            f"final lineage={row['final_lineage']:>4}  |  "
            f"ratio={row['ratio']:.3f}"
        )

    # -- Plotly visualizations ---------------------------------------------
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("\nInstall plotly for visualization: pip install plotly")
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scatter3d"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        subplot_titles=[
            "Pareto Front (DTLZ2)",
            "Operator Effectiveness",
            "Generation Contribution Ratio",
            "Final Lineage per Generation",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # 1. 3D Pareto front: the final set of non-dominated solutions.
    fig.add_trace(
        go.Scatter3d(
            x=F[:, 0],
            y=F[:, 1],
            z=F[:, 2],
            mode="markers",
            marker=dict(size=3, color=F[:, 2], colorscale="Viridis", opacity=0.8),
            name="Pareto front",
        ),
        row=1,
        col=1,
    )

    # 2. Operator bar chart: compare total uses vs. uses that ended up in
    #    the ancestry of the final front. Operators with a higher "In final
    #    lineages" bar relative to "Total uses" were more productive.
    op_names = [str(r["operator"]) for r in operator_stats]
    op_total = [int(r["total_uses"]) for r in operator_stats]
    op_final = [int(r["uses_in_final_lineages"]) for r in operator_stats]

    fig.add_trace(
        go.Bar(
            x=op_names,
            y=op_total,
            name="Total uses",
            marker_color="lightblue",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=op_names,
            y=op_final,
            name="In final lineages",
            marker_color="steelblue",
        ),
        row=1,
        col=2,
    )

    # 3. Contribution ratio over generations: what fraction of individuals
    #    created in each generation became ancestors of the final front.
    #    Early generations often have high ratios (founding genetic material),
    #    later generations refine solutions closer to the front.
    gens = [int(r["generation"]) for r in gen_contributions]
    ratios = [float(r["ratio"]) for r in gen_contributions]

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=ratios,
            mode="lines+markers",
            marker=dict(size=4, color="steelblue"),
            line=dict(width=1.5),
            name="Contribution ratio",
        ),
        row=2,
        col=1,
    )

    # 4. Absolute count of individuals per generation that are in the final
    #    front's lineage. Spikes indicate generations where particularly
    #    useful genetic material was introduced.
    gen_final = [int(r["final_lineage"]) for r in gen_contributions]

    fig.add_trace(
        go.Bar(
            x=gens,
            y=gen_final,
            name="Final lineage count",
            marker_color="steelblue",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text="NSGA-II Genealogy Analysis - DTLZ2",
        height=850,
        width=1200,
        showlegend=False,
    )
    fig.update_scenes(
        xaxis_title="f1",
        yaxis_title="f2",
        zaxis_title="f3",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Operator", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Generation", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.show()
    print("\nPlotly dashboard opened in browser.")


if __name__ == "__main__":
    main()

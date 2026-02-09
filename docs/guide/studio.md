# VAMOS Studio

VAMOS Studio is an interactive dashboard for exploring optimization results, visualizing Pareto fronts, and performing sensitivity analysis. It behaves as a companion app to your experiment results.

## Launching Studio

To start the studio, ensure you have the `studio` extra installed (`pip install -e ".[studio]"`), then point it to your results directory:

```bash
vamos studio --study-dir results/
```

Or via python module:

```bash
python -m vamos.studio.main --study-dir results/
```

By default, it will serve on `http://localhost:8501` (Streamlit default).

## Features

### Study Explorer
- **Overview**: See all algorithms, problems, and seeds in your results folder.
- **Filtering**: Select specific subsets of data to visualize.

### Visualization
- **Pareto Fronts**: Interactive 2D and 3D scatter plots of the approximation sets.
- **Parallel Coordinates**: Visualize high-dimensional solution vectors.
- **Convergence**: View hypervolume (HV) and IGD metrics over evaluations (if quality-over-time data is available).

### Analysis
- **MCDM**: Apply decision-making preferences (weights, reference points) to select "best" solutions interactively. Supports **Weighted Sum**, **Tchebycheff**, **Knee Point**, and **TOPSIS**.
- **Comparison**: Side-by-side comparison of different algorithms. Overlays Pareto fronts to visually assess dominance and diversity.
- **Search Dynamics**: Re-run experiments directly from the Studio to visualize the population evolution over generations (animated scatter plot).
- **Landscape Analysis**: Run a random walk analysis on the selected problem to visualize fitness landscape roughness via autocorrelation plots.

## Requirements
Studio relies on `streamlit`, `plotly`, and `pandas`. These are pulled in via the `studio` optional dependency group.

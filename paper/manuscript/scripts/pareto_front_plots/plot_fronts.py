"""
Generate four publication-quality Pareto front plots for NSGA-II variants
on ZDT4 (2-obj) and DTLZ2 (3-obj).

Plots:
  1. ZDT4  – standard NSGA-II
  2. ZDT4  – steady-state NSGA-II
  3. DTLZ2 – standard NSGA-II
  4. DTLZ2 – unbounded-archive NSGA-II

Output: PNG files saved to ../../figures/ (project root figures folder).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D projection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIG_DIR = SCRIPT_DIR.parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colour palette
REF_COLOR = "#4A4A4A"
FRONT_COLOR = "#E74C3C"
FRONT_COLOR_ALT = "#2E86C1"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> np.ndarray:
    """Load a headerless, comma-separated CSV from DATA_DIR."""
    return np.loadtxt(DATA_DIR / filename, delimiter=",")


# ---------------------------------------------------------------------------
# 2-D plot (ZDT4)
# ---------------------------------------------------------------------------

def plot_zdt4(obtained_file: str, title: str, out_name: str) -> None:
    """Create a 2-D Pareto-front plot for ZDT4."""
    ref = load_csv("ZDT4.csv")
    obt = load_csv(obtained_file)

    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    # Reference front (line)
    idx = np.argsort(ref[:, 0])
    ax.plot(
        ref[idx, 0], ref[idx, 1],
        color=REF_COLOR, linewidth=1.2, label="Reference front", zorder=2,
    )

    # Obtained front (scatter)
    ax.scatter(
        obt[:, 0], obt[:, 1],
        s=9, color="black", edgecolors="white", linewidths=0.2,
        alpha=0.85, label="NSGA-II", zorder=3,
    )

    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    print(f"  ✓ {out_name}")


# ---------------------------------------------------------------------------
# 3-D plot (DTLZ2)
# ---------------------------------------------------------------------------

def plot_dtlz2(obtained_file: str, title: str, out_name: str) -> None:
    """Create a 3-D Pareto-front plot for DTLZ2."""
    ref = load_csv("DTLZ2.3D.csv")
    obt = load_csv(obtained_file)

    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111, projection="3d")

    # Reference front (semi-transparent surface-like scatter)
    ax.scatter(
        ref[:, 0], ref[:, 1], ref[:, 2],
        s=1, color=REF_COLOR, alpha=0.12, label="Reference front", zorder=1,
    )

    # Obtained front (scatter)
    ax.scatter(
        obt[:, 0], obt[:, 1], obt[:, 2],
        s=10, color="black", edgecolors="white", linewidths=0.2,
        alpha=0.9, label="NSGA-II", zorder=3,
    )

    ax.set_xlabel("$f_1$", labelpad=4)
    ax.set_ylabel("$f_2$", labelpad=4)
    ax.set_zlabel("$f_3$", labelpad=4)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 0.95), framealpha=0.9)

    ax.view_init(elev=35, azim=45)
    ax.tick_params(axis="both", labelsize=7)

    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)
    print(f"  ✓ {out_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating Pareto front plots …")

    # ZDT4
    plot_zdt4(
        "FUN.NSGAII.ZDT4.csv",
        "ZDT4 — Standard NSGA-II",
        "front_zdt4_standard.png",
    )
    plot_zdt4(
        "FUN.NSGAII.ZDT4.steady_state.csv",
        "ZDT4 — Steady-State NSGA-II",
        "front_zdt4_steady_state.png",
    )

    # DTLZ2
    plot_dtlz2(
        "FUN.NSGAII.DTLZ2.csv",
        "DTLZ2 — Standard NSGA-II",
        "front_dtlz2_standard.png",
    )
    plot_dtlz2(
        "FUN.NSGAII.DTLZ2.archive.csv",
        "DTLZ2 — Archive NSGA-II",
        "front_dtlz2_archive.png",
    )

    print(f"Done. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

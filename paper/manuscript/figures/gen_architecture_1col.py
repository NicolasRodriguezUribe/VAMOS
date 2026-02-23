"""Generate a single-column architecture diagram for VAMOS.

Matches the style of the original two-column figure:
- Rounded boxes with light-gray fill and subtle border
- Teal header text with a small icon to the left
- Gray body text
- Triple-chevron arrows between layers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

# IEEE single column ~3.5 in.  Slightly taller to accommodate icons.
fig, ax = plt.subplots(figsize=(3.4, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.5)
ax.axis("off")

# --- palette (taken from the original figure) ---
TEAL = "#00838F"
BOX_BG = "#F7F7F7"
BORDER = "#C8CDD0"
HEADER_FG = TEAL
BODY_FG = "#555555"

# --- layer data ---
layers = [
    {
        "title": "UX Layer",
        "items": [
            "VAMOS Studio \u00b7 Problem Builder",
            "Results Explorer \u00b7 Onboarding",
        ],
    },
    {
        "title": "Experiment Layer",
        "items": [
            "CLI Tools \u00b7 Benchmarking",
            "Table/Plot Generation \u00b7 Results Analysis",
        ],
    },
    {
        "title": "Engine Layer",
        "items": [
            "NSGA-II \u00b7 NSGA-III \u00b7 MOEA/D",
            "SMS-EMOA \u00b7 SPEA2 \u00b7 IBEA",
            "SMPSO \u00b7 AGE-MOEA \u00b7 RVEA",
        ],
    },
    {
        "title": "Foundation Layer",
        "items": [
            "Problems \u00b7 Kernels",
            "Metrics \u00b7 Archives",
        ],
    },
]

# --- layout constants ---
box_w = 9.0
box_x = (10 - box_w) / 2
header_h = 0.50
gap = 0.55
line_h = 0.36
body_pad = 0.25
icon_x = box_x + 0.55  # icon centre x
text_cx = 5.3          # body/header text centre (shifted right to leave room for icon)


def layer_height(layer):
    return header_h + 2 * body_pad + len(layer["items"]) * line_h


# ── icon drawing helpers ────────────────────────────────────────────────

def draw_monitor(ax, cx, cy, s=0.18):
    """Small monitor / dashboard icon."""
    # screen
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - s, cy - s * 0.6), 2 * s, s * 1.2,
        boxstyle="round,pad=0.02", fc=TEAL, ec=TEAL, lw=0.6))
    # stand
    ax.plot([cx, cx], [cy - s * 0.6, cy - s * 1.0], color=TEAL, lw=1.0)
    ax.plot([cx - s * 0.5, cx + s * 0.5], [cy - s * 1.0, cy - s * 1.0],
            color=TEAL, lw=1.0)


def draw_flask(ax, cx, cy, s=0.18):
    """Small flask / beaker icon."""
    # neck
    ax.plot([cx - s * 0.25, cx - s * 0.25], [cy + s * 0.9, cy + s * 0.2],
            color=TEAL, lw=1.0)
    ax.plot([cx + s * 0.25, cx + s * 0.25], [cy + s * 0.9, cy + s * 0.2],
            color=TEAL, lw=1.0)
    # body (triangle)
    tri_x = [cx - s * 0.8, cx + s * 0.8, cx + s * 0.25, cx - s * 0.25, cx - s * 0.8]
    tri_y = [cy - s * 0.9, cy - s * 0.9, cy + s * 0.2, cy + s * 0.2, cy - s * 0.9]
    ax.fill(tri_x, tri_y, fc=TEAL, ec=TEAL, lw=0.6, alpha=0.85)
    # rim
    ax.plot([cx - s * 0.35, cx + s * 0.35], [cy + s * 0.9, cy + s * 0.9],
            color=TEAL, lw=1.2)


def draw_gear(ax, cx, cy, s=0.18):
    """Small gear / cog icon."""
    r_outer = s * 0.95
    r_inner = s * 0.55
    n_teeth = 8
    angles = np.linspace(0, 2 * np.pi, n_teeth * 2, endpoint=False)
    xs, ys = [], []
    for k, a in enumerate(angles):
        r = r_outer if k % 2 == 0 else r_inner
        xs.append(cx + r * np.cos(a))
        ys.append(cy + r * np.sin(a))
    xs.append(xs[0])
    ys.append(ys[0])
    ax.fill(xs, ys, fc=TEAL, ec=TEAL, lw=0.5)
    # centre hole
    circle = plt.Circle((cx, cy), s * 0.2, fc="white", ec="white", lw=0)
    ax.add_patch(circle)


def draw_blocks(ax, cx, cy, s=0.18):
    """Small stacked-blocks / foundation icon."""
    bw, bh = s * 0.7, s * 0.45
    gap_b = s * 0.08
    # bottom row: two blocks
    for dx in [-bw - gap_b / 2, gap_b / 2]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx + dx - bw * 0.15, cy - bh - gap_b),
            bw, bh, boxstyle="round,pad=0.01",
            fc=TEAL, ec=TEAL, lw=0.5))
    # top row: one block centred
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - bw / 2, cy + gap_b * 0.5),
        bw, bh, boxstyle="round,pad=0.01",
        fc=TEAL, ec=TEAL, lw=0.5))


icon_funcs = [draw_monitor, draw_flask, draw_gear, draw_blocks]

# ── chevron arrows ──────────────────────────────────────────────────────

def draw_chevrons(ax, cx, y_top, y_bot, n=3):
    """Draw *n* small downward chevrons between y_top and y_bot."""
    chev_h = 0.09
    chev_w = 0.15
    spacing = (y_top - y_bot) / (n + 1)
    for k in range(1, n + 1):
        yc = y_top - k * spacing
        xs = [cx - chev_w, cx, cx + chev_w]
        ys = [yc + chev_h, yc - chev_h, yc + chev_h]
        ax.plot(xs, ys, color=TEAL, lw=1.4, solid_capstyle="round")


# ── draw layers ─────────────────────────────────────────────────────────

y_cursor = 12.5

for i, layer in enumerate(layers):
    h = layer_height(layer)

    # chevron arrows between layers
    if i > 0:
        draw_chevrons(ax, 5.0, y_cursor + gap, y_cursor + 0.02)

    # outer rounded box
    outer = mpatches.FancyBboxPatch(
        (box_x, y_cursor - h), box_w, h,
        boxstyle="round,pad=0.15",
        facecolor=BOX_BG, edgecolor=BORDER, linewidth=1.0)
    ax.add_patch(outer)

    # header text (icon + title)
    header_cy = y_cursor - header_h / 2
    icon_funcs[i](ax, icon_x, header_cy, s=0.17)
    ax.text(
        text_cx, header_cy, layer["title"],
        ha="center", va="center",
        fontsize=8, fontweight="bold",
        color=HEADER_FG, family="sans-serif")

    # thin separator line below header
    sep_y = y_cursor - header_h
    ax.plot([box_x + 0.3, box_x + box_w - 0.3], [sep_y, sep_y],
            color=BORDER, lw=0.6)

    # body text
    body_top = sep_y - body_pad
    for j, line in enumerate(layer["items"]):
        ax.text(
            5.0, body_top - j * line_h, line,
            ha="center", va="center",
            fontsize=6.5, color=BODY_FG, family="sans-serif")

    y_cursor -= h + gap

# title
ax.text(
    5.0, 13.15, "VAMOS Architecture",
    ha="center", va="center",
    fontsize=11, fontweight="bold",
    color="#212121", family="sans-serif")

plt.tight_layout(pad=0.2)
out = "paper/manuscript/figures/architecture_1col.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()

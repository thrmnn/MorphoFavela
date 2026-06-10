#!/usr/bin/env python3
"""BRISA slide deck — fig_facade_concept_schematic.png.

Pure schematic. 2-3 buildings of varying heights on a slope, a low northern
winter-solstice sun (Rio is at ~23 deg S), with facade panels coloured by
expected direct-sun-hours: N facades dark (sunlit), S facades bright (shaded),
E/W intermediate. The point is to show that the same ray-cast engine that
drives BRISA at ground level can lift onto building skins, producing one
diagnostic value per facade panel.

Note: in the Southern Hemisphere, winter sun is in the NORTH sky — so
N-facing facades receive direct sun, S-facing facades stay shaded.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_facade_concept_schematic.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from fig_style import apply_style  # noqa: E402

OUT = Path("/home/theo/brisa_paper/artifacts/slides/assets/fig_facade_concept_schematic.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Facade colours encode expected winter direct-sun hours
# (using the same RdYlBu_r-style sense as the deck's risk encoding,
# but flipped: bright/yellow = SUN, dark = SHADE — easier to read on slides)
SUN_COL = "#f5b942"     # sunlit (N facade)
PARTIAL_COL = "#c9a567" # partial (E facade — morning sun only)
PARTIAL2_COL = "#7a6a55" # partial (W facade — afternoon sun)
SHADE_COL = "#2a2a35"   # shaded (S facade)
BLDG_OUTLINE = "#111111"
GROUND = "#d8d4cc"
SKY = "#fbf9f5"
SUN_YELLOW = "#f0c64a"


def building_box(ax, x0, base_y, w, h, label=None,
                 sun_col=SUN_COL, shade_col=SHADE_COL,
                 left_col=PARTIAL_COL, right_col=PARTIAL2_COL,
                 n_floors=None):
    """Draw an axonometric-ish 2.5D box with three visible faces.

    Front face (faces viewer, treated as N-facing in southern hemisphere)
    is the SUNLIT one. Top is roof. Right face is one of the lateral E/W
    facades and gets partial sun. The hidden back (S) face is also drawn
    as a stripe at far edge to make the four-orientation point readable.
    """
    # Iso depth
    d = w * 0.32
    dy = h * 0.18

    # FRONT (N) face — main rectangle facing viewer
    front = Polygon(
        [(x0, base_y), (x0 + w, base_y), (x0 + w, base_y + h), (x0, base_y + h)],
        closed=True, facecolor=sun_col, edgecolor=BLDG_OUTLINE, linewidth=1.0, zorder=4,
    )
    ax.add_patch(front)

    # RIGHT (E or W) face — slanted to the right & up
    right = Polygon(
        [
            (x0 + w, base_y),
            (x0 + w + d, base_y + dy),
            (x0 + w + d, base_y + h + dy),
            (x0 + w, base_y + h),
        ],
        closed=True, facecolor=right_col, edgecolor=BLDG_OUTLINE, linewidth=1.0, zorder=4,
    )
    ax.add_patch(right)

    # ROOF
    roof = Polygon(
        [
            (x0, base_y + h),
            (x0 + w, base_y + h),
            (x0 + w + d, base_y + h + dy),
            (x0 + d, base_y + h + dy),
        ],
        closed=True, facecolor="#efeae0", edgecolor=BLDG_OUTLINE, linewidth=1.0, zorder=5,
    )
    ax.add_patch(roof)

    # Floor lines on the front face (the "panel" idea per floor)
    if n_floors is None:
        n_floors = max(2, int(round(h / 2.0)))
    for k in range(1, n_floors):
        y = base_y + h * k / n_floors
        ax.plot([x0, x0 + w], [y, y], color=BLDG_OUTLINE, lw=0.5, alpha=0.45, zorder=5)
        # mirror onto the slanted right face
        ax.plot([x0 + w, x0 + w + d], [y, y + dy], color=BLDG_OUTLINE, lw=0.5, alpha=0.45, zorder=5)

    # Vertical separators on the front face (panel width ~ w/3)
    n_cols = max(2, int(round(w / 2.5)))
    for c in range(1, n_cols):
        x = x0 + w * c / n_cols
        ax.plot([x, x], [base_y, base_y + h], color=BLDG_OUTLINE, lw=0.4, alpha=0.4, zorder=5)

    # BACK (S) face as a thin stripe peeking past the right edge to remind
    # the viewer there is a shaded back facade. We draw it as a darker
    # vertical strip behind the right face.
    back_x = x0 + w + d * 0.6
    back = Polygon(
        [
            (back_x, base_y + dy * 0.6),
            (back_x + w * 0.18, base_y + dy * 0.6),
            (back_x + w * 0.18, base_y + h + dy * 0.6),
            (back_x, base_y + h + dy * 0.6),
        ],
        closed=True, facecolor=shade_col, edgecolor=BLDG_OUTLINE, linewidth=0.6, zorder=3,
    )
    ax.add_patch(back)

    if label:
        ax.text(
            x0 + w / 2, base_y + h + dy + 0.6, label,
            ha="center", va="bottom", fontsize=12, color="#444",
        )


def draw_sun(ax, cx, cy, r=1.6):
    sun = plt.Circle((cx, cy), r, facecolor=SUN_YELLOW, edgecolor="#b9923a",
                     linewidth=1.0, zorder=2)
    ax.add_patch(sun)
    # rays
    for k in range(12):
        a = k * np.pi / 6
        x1 = cx + np.cos(a) * r * 1.15
        y1 = cy + np.sin(a) * r * 1.15
        x2 = cx + np.cos(a) * r * 1.55
        y2 = cy + np.sin(a) * r * 1.55
        ax.plot([x1, x2], [y1, y2], color="#b9923a", lw=1.1, zorder=2)


def draw_ray(ax, x0, y0, x1, y1, color="#b9923a", lw=1.2):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )


def main():
    apply_style()

    fig_w, fig_h = 16.0, 9.0  # 16:9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.set_facecolor(SKY)
    fig.patch.set_facecolor(SKY)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # SLOPE — a hillside rising left-to-right.
    slope_pts = [(0, 1.8), (8, 2.4), (15, 3.6), (22, 5.2), (32, 7.0), (32, 0), (0, 0)]
    slope = Polygon(slope_pts, closed=True, facecolor=GROUND,
                    edgecolor="#aaa49a", linewidth=0.8, zorder=1)
    ax.add_patch(slope)

    # Three buildings, set onto the slope at rising bases.
    # Building footprint widths and heights chosen to read clearly at slide scale.
    b1 = dict(x0=3.0, base_y=2.4, w=4.2, h=4.5)   # 2 floors
    b2 = dict(x0=11.5, base_y=3.1, w=4.6, h=7.2)  # 3 floors
    b3 = dict(x0=20.0, base_y=4.8, w=5.2, h=9.0)  # 4 floors

    building_box(ax, **b1, n_floors=2)
    building_box(ax, **b2, n_floors=3)
    building_box(ax, **b3, n_floors=4)

    # SUN — upper-right but north-side (Rio: winter sun in N sky, low altitude)
    sun_cx, sun_cy = 28.5, 15.2
    draw_sun(ax, sun_cx, sun_cy, r=1.0)
    ax.text(sun_cx, sun_cy + 1.8, "winter sun  (low, north)",
            ha="center", va="bottom", fontsize=13, color="#6a4f1a", style="italic")

    # A handful of rays from the sun grazing the tallest building, illustrating
    # the ray-cast that produces the per-panel score. Hit the front face only
    # to show "this face gets sun".
    for fy in (b3["base_y"] + 1.5, b3["base_y"] + 4.5, b3["base_y"] + 7.2):
        draw_ray(ax, sun_cx - 1.1, sun_cy - 0.6, b3["x0"] + b3["w"] * 0.85, fy,
                 color="#d2a437", lw=1.0)

    # One ray that GRAZES the top of b2 and casts a shadow onto b3's lower front
    draw_ray(ax, sun_cx - 1.2, sun_cy - 0.9, b2["x0"] + b2["w"] + 0.4, b2["base_y"] + b2["h"] + 0.2,
             color="#d2a437", lw=0.8)

    # Callout: "one diagnostic value per facade panel"
    # Point an arrow at one specific panel on b3.
    panel_cx = b3["x0"] + b3["w"] * 0.5 / 2 + b3["w"] * 0.25
    panel_cy = b3["base_y"] + b3["h"] * (0.5 + 0.5) / 4  # mid-height
    # Highlight one panel with a thicker outline.
    px0 = b3["x0"] + b3["w"] * 1 / 3
    px1 = b3["x0"] + b3["w"] * 2 / 3
    py0 = b3["base_y"] + b3["h"] * 2 / 4
    py1 = b3["base_y"] + b3["h"] * 3 / 4
    ax.add_patch(Polygon(
        [(px0, py0), (px1, py0), (px1, py1), (px0, py1)],
        closed=True, facecolor=SUN_COL, edgecolor="#c0392b", linewidth=2.0, zorder=6,
    ))
    callout_x, callout_y = 16.5, 14.0
    ax.annotate(
        "one diagnostic value\nper facade panel",
        xy=((px0 + px1) / 2, (py0 + py1) / 2),
        xytext=(callout_x, callout_y),
        ha="center", va="center", fontsize=13.5, color="#222",
        arrowprops=dict(arrowstyle="-", color="#444", lw=0.9,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999", linewidth=0.6),
        zorder=7,
    )

    # Legend strip — bottom-left — showing the 4 orientation colours
    leg_x0 = 0.8
    leg_y0 = 0.7
    sw = 1.0
    sh = 0.7
    pad = 0.25
    orientations = [
        ("N (front)", SUN_COL),
        ("E (morning)", PARTIAL_COL),
        ("W (afternoon)", PARTIAL2_COL),
        ("S (back)", SHADE_COL),
    ]
    for i, (lab, col) in enumerate(orientations):
        x = leg_x0 + i * (sw + 2.4)
        ax.add_patch(Polygon(
            [(x, leg_y0), (x + sw, leg_y0), (x + sw, leg_y0 + sh), (x, leg_y0 + sh)],
            closed=True, facecolor=col, edgecolor=BLDG_OUTLINE, linewidth=0.7, zorder=8,
        ))
        ax.text(x + sw + pad, leg_y0 + sh / 2, lab,
                va="center", ha="left", fontsize=11, color="#222")

    # Small footer note pinning the geography
    ax.text(
        31.5, 0.4,
        "Rio de Janeiro · winter solstice · sun azimuth from N quadrant",
        ha="right", va="bottom", fontsize=9.5, color="#777", style="italic",
    )

    plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005)
    fig.savefig(OUT, dpi=300, facecolor=SKY)
    plt.close(fig)
    print(f"Saved {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

"""Line diagrams of the three ventilation measures, for the report: frontal
area density (left), canyon alignment (centre), upwind shelter angle (right)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.om_package.fig_style import DPI, FONT_PT, LINE_DARK, TEXT_WIDTH_IN, figure_style, save

FIG_HEIGHT_IN = 1.95
WALL_FACING = "#E69F00"
BUILDING_FACE = "#ececec"
BUILDING_SIDE = "#d2d2d2"
GROUND = "#f4f1e8"
EDGE = "#7a7a7a"
#: Oblique projection: depth axis drawn up and to the right.
_DEPTH = np.array([0.55, 0.38])


def _p(x, y, z):
    """Oblique projection of (x across the wind, y downwind depth, z up)."""
    return np.array([x + _DEPTH[0] * y, z + _DEPTH[1] * y])


def _poly(ax, pts, face, edge=EDGE, lw=0.7, z=2):
    from matplotlib.patches import Polygon

    ax.add_patch(Polygon(np.array(pts), closed=True, facecolor=face, edgecolor=edge, lw=lw, zorder=z))


def _arrow(ax, start, end, colour=LINE_DARK, lw=1.4, z=6, halo=False):
    import matplotlib.patheffects as mpe

    a = ax.annotate("", xy=end, xytext=start, zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, mutation_scale=10, shrinkA=0, shrinkB=0))
    if halo:
        a.arrow_patch.set_path_effects([mpe.Stroke(linewidth=lw + 2.6, foreground="white"), mpe.Normal()])


def _box(ax, x0, y0, w, d, h):
    """Box with its front (windward, y = y0) face highlighted."""
    _poly(ax, [_p(x0 + w, y0, 0), _p(x0 + w, y0 + d, 0), _p(x0 + w, y0 + d, h), _p(x0 + w, y0, h)], BUILDING_SIDE, z=3)
    _poly(ax, [_p(x0, y0, h), _p(x0 + w, y0, h), _p(x0 + w, y0 + d, h), _p(x0, y0 + d, h)], BUILDING_FACE, z=3)
    _poly(ax, [_p(x0, y0, 0), _p(x0 + w, y0, 0), _p(x0 + w, y0, h), _p(x0, y0, h)], WALL_FACING, edge="#9a6500", z=4)


def _frontal(ax):
    L = 4.0
    _poly(ax, [_p(0, 0, 0), _p(L, 0, 0), _p(L, L, 0), _p(0, L, 0)], GROUND, z=1)
    _box(ax, 1.9, 2.3, 1.5, 1.2, 1.9)
    _box(ax, 0.4, 0.6, 1.3, 1.2, 2.4)
    _arrow(ax, _p(2.6, -2.6, 0), _p(2.6, -0.5, 0))
    ax.text(*(_p(2.6, -2.6, 0) + [0.25, -0.1]), "wind", ha="left", va="center", fontsize=FONT_PT)
    ax.text(*_p(3.15, 0.45, 0), "ground\narea", ha="center", va="center", fontsize=FONT_PT, linespacing=0.95,
            zorder=5)
    ax.text(*(_p(0, L, 0) + [-0.55, 2.1]), "wall area facing the wind", ha="left", va="bottom",
            fontsize=FONT_PT, color="#9a6500")
    ax.text(*(_p(0, L, 0) + [-0.55, 1.6]), "divided by ground area", ha="left", va="bottom", fontsize=FONT_PT)
    ax.set_xlim(-0.8, 6.5)
    ax.set_ylim(-1.3, 5.0)


def _alignment(ax):
    from matplotlib.patches import FancyBboxPatch

    width = 2.2
    for y0 in (width / 2, -width / 2 - 1.3):
        for x0 in (-4.0, -1.35, 1.3):
            ax.add_patch(FancyBboxPatch((x0, y0), 2.4, 1.3, boxstyle="square,pad=0", facecolor=BUILDING_FACE,
                                        edgecolor=EDGE, lw=0.7, zorder=2))
    ax.plot([-4.3, 4.0], [0, 0], color=EDGE, lw=0.6, ls=(0, (1.5, 2.5)), zorder=1)
    _arrow(ax, (-3.6, -0.35), (-0.9, -0.35))
    ax.text(-2.25, 0.05, "0°: along the street", ha="center", va="bottom", fontsize=FONT_PT)
    _arrow(ax, (2.4, -2.6), (2.4, 2.6), halo=True)
    ax.text(2.4, 2.75, "90°: across", ha="center", va="bottom", fontsize=FONT_PT)
    ax.text(-0.15, 3.25, "plan view", ha="center", va="bottom", fontsize=FONT_PT, color="#555555")
    ax.set_xlim(-4.4, 4.1)
    ax.set_ylim(-3.0, 4.1)


def _person(ax, x, eye):
    head_r = 0.17
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
        (x, eye), head_r, facecolor="white", edgecolor=LINE_DARK, lw=0.9, zorder=5))
    neck, hip = eye - head_r, eye - 0.85
    ax.plot([x, x], [neck, hip], color=LINE_DARK, lw=0.9, zorder=5)
    ax.plot([x - 0.25, x, x + 0.25], [0, hip, 0], color=LINE_DARK, lw=0.9, zorder=5)
    ax.plot([x - 0.3, x, x + 0.3], [hip + 0.05, neck - 0.15, hip + 0.05], color=LINE_DARK, lw=0.9, zorder=5)


def _shelter(ax):
    ground = -0.6
    ax.plot([-0.4, 7.4], [0, 0], color=LINE_DARK, lw=0.9, zorder=2)
    _poly(ax, [(0.2, 0), (2.4, 0), (2.4, 3.6), (0.2, 3.6)], BUILDING_FACE, z=2)
    _poly(ax, [(6.2, 0), (7.4, 0), (7.4, 2.4), (6.2, 2.4)], BUILDING_FACE, z=2)
    eye_x, eye = 4.9, 1.5
    _person(ax, eye_x, eye)
    top = (2.4, 3.6)
    ax.plot([eye_x, top[0]], [eye, top[1]], color=WALL_FACING, lw=1.1, zorder=4)
    ax.plot([eye_x, 2.6], [eye, eye], color=EDGE, lw=0.7, ls=(0, (2, 2)), zorder=4)
    ang = math.degrees(math.atan2(top[1] - eye, eye_x - top[0]))
    r = 1.25
    t = np.radians(np.linspace(180 - ang, 180, 40))
    ax.plot(eye_x + r * np.cos(t), eye + r * np.sin(t), color=WALL_FACING, lw=1.0, zorder=4)
    ax.text(eye_x - 0.55, eye - 0.15, "upwind\nshelter angle", ha="right", va="top", fontsize=FONT_PT,
            color="#9a6500", linespacing=0.95)
    _arrow(ax, (-0.4, 4.35), (1.6, 4.35))
    ax.text(0.6, 4.5, "wind", ha="center", va="bottom", fontsize=FONT_PT)
    ax.text(eye_x, ground, "street", ha="center", va="center", fontsize=FONT_PT)
    ax.text(1.3, ground, "upwind building", ha="center", va="center", fontsize=FONT_PT)
    ax.set_xlim(-0.5, 7.6)
    ax.set_ylim(-0.95, 5.2)


def build_fig_vent_schematic(out_path) -> Path:
    import matplotlib.pyplot as plt

    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI,
                                 gridspec_kw={"width_ratios": [1.0, 1.0, 1.05], "wspace": 0.08})
        for ax, draw in zip(axes, (_frontal, _alignment, _shelter)):
            draw(ax)
            ax.set_aspect("equal")
            ax.set_axis_off()
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
        return save(fig, Path(out_path))

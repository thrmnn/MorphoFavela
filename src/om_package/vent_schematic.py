"""Line diagrams for the report: the three ventilation measures computed in the
package (top row) and the three street flow regimes of Oke (1988) that the
height-to-width ratio distinguishes (bottom row).

Everything is drawn on one axes in inches, so 8 pt text is 8 pt at print size."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.om_package.fig_style import DPI, FONT_PT, LINE_DARK, TEXT_WIDTH_IN, figure_style, save

FIG_HEIGHT_IN = 3.85
WALL_FACING = "#E69F00"
WALL_EDGE = "#9a6500"
BUILDING_FACE = "#ececec"
BUILDING_SIDE = "#d2d2d2"
GROUND = "#f4f1e8"
EDGE = "#7a7a7a"
WIND = "#222222"
WIND_LW = 1.3
#: Literature thresholds on the height-to-width ratio (Oke, 1988).
RATIO_ISOLATED_MAX = 0.35
RATIO_SKIMMING_MIN = 0.65
#: Oblique projection: depth axis drawn up and to the right (inches per unit).
_DEPTH = np.array([0.5, 0.33])
_TOP_BASE = 2.07
_ROW_TOP = 3.6


def _txt(ax, x, y, s, **kw):
    kw.setdefault("fontsize", FONT_PT)
    kw.setdefault("color", LINE_DARK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, **kw)


def _poly(ax, pts, face, edge=EDGE, lw=0.7, z=2):
    from matplotlib.patches import Polygon

    ax.add_patch(Polygon(np.array(pts), closed=True, facecolor=face, edgecolor=edge, lw=lw, zorder=z))


def _rect(ax, x0, y0, x1, y1, face=BUILDING_FACE, z=2):
    _poly(ax, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], face, z=z)


def _head(ax, p0, p1):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=WIND, lw=WIND_LW,
                                 shrinkA=0, shrinkB=0, zorder=6))


def _arrow(ax, start, end):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color=WIND, lw=WIND_LW,
                                 shrinkA=0, shrinkB=0, zorder=6))


def _stream(ax, xs, ys, n=80):
    """Smooth streamline through the points, ending in the same arrowhead as the wind arrows."""
    from scipy.interpolate import PchipInterpolator

    t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    tt = np.linspace(0, t[-1], n)
    x, y = PchipInterpolator(t, xs)(tt), PchipInterpolator(t, ys)(tt)
    ax.plot(x[:-2], y[:-2], color=WIND, lw=WIND_LW, solid_capstyle="butt", zorder=5)
    _head(ax, (x[-4], y[-4]), (x[-1], y[-1]))


def _loop(ax, cx, cy, rx, ry, a0=200, a1=-110):
    """Clockwise closed vortex: an elliptical arc with an arrowhead."""
    t = np.radians(np.linspace(a0, a1, 60))
    x, y = cx + rx * np.cos(t), cy + ry * np.sin(t)
    ax.plot(x[:-3], y[:-3], color=WIND, lw=WIND_LW, solid_capstyle="butt", zorder=5)
    _head(ax, (x[-5], y[-5]), (x[-1], y[-1]))


def _header(ax, x, letter, title):
    t = _txt(ax, x, _ROW_TOP + 0.15, letter, fontweight="bold")
    _txt(ax, x + 0.14, _ROW_TOP + 0.15, title)
    return t


def _frontal(ax, ox=0.05, oy=2.74):
    def p(x, y, z):
        return (ox + x + _DEPTH[0] * y, oy + z + _DEPTH[1] * y)

    side = 1.15
    _poly(ax, [p(0, 0, 0), p(side, 0, 0), p(side, side, 0), p(0, side, 0)], GROUND, z=1)

    def box(x0, y0, w, d, h):
        _poly(ax, [p(x0 + w, y0, 0), p(x0 + w, y0 + d, 0), p(x0 + w, y0 + d, h), p(x0 + w, y0, h)],
              BUILDING_SIDE, z=3)
        _poly(ax, [p(x0, y0, h), p(x0 + w, y0, h), p(x0 + w, y0 + d, h), p(x0, y0 + d, h)], BUILDING_FACE, z=3)
        _poly(ax, [p(x0, y0, 0), p(x0 + w, y0, 0), p(x0 + w, y0, h), p(x0, y0, h)], WALL_FACING, edge=WALL_EDGE, z=4)

    box(0.12, 0.12, 0.38, 0.3, 0.6)
    box(0.62, 0.6, 0.38, 0.3, 0.5)
    a0, a1 = p(0.55, -0.85, 0), p(0.55, -0.1, 0)
    _arrow(ax, a0, a1)
    _txt(ax, a0[0] + 0.1, a0[1] - 0.08, "wind")
    _txt(ax, ox, _TOP_BASE + 0.05, "orange wall area ÷ ground area")


def _alignment(ax, x0=1.95, width=2.1):
    street_lo, street_hi, depth = 2.7, 3.3, 0.3
    rows = {street_lo - depth: (street_lo - depth, street_lo), street_hi: (street_hi, street_hi + depth)}
    for ylo, yhi in rows.values():
        for a, b in ((0.0, 0.6), (0.7, 1.3), (1.6, width)):
            _rect(ax, x0 + a, ylo, x0 + b, yhi)
    mid = (street_lo + street_hi) / 2

    _arrow(ax, (x0 + 0.02, street_lo + 0.14), (x0 + 0.55, street_lo + 0.14))
    _txt(ax, x0 + 0.02, street_hi - 0.17, "0°: along")

    sx, sy, ang, ln = x0 + 0.7, street_lo + 0.14, 40, 0.58
    ex, ey = sx + ln * math.cos(math.radians(ang)), sy + ln * math.sin(math.radians(ang))
    ax.plot([sx, sx + 0.58], [sy, sy], color=EDGE, lw=0.7, ls=(0, (2, 2)), zorder=4)
    _arrow(ax, (sx, sy), (ex, ey))
    t = np.radians(np.linspace(0, ang, 30))
    ax.plot(sx + 0.3 * np.cos(t), sy + 0.3 * np.sin(t), color=WALL_FACING, lw=1.3, zorder=5)

    gx = x0 + 1.45
    _arrow(ax, (gx, street_lo - depth - 0.1), (gx, street_hi + depth + 0.05))
    _txt(ax, gx + 0.1, mid, "90°: across")
    _txt(ax, sx + 0.33, sy + 0.08, "40\u00b0")
    _txt(ax, x0 - 0.02, _TOP_BASE + 0.05, "angle between street and wind")


def _person(ax, x, g, eye):
    from matplotlib.patches import Circle

    r = 0.06
    ax.add_patch(Circle((x, eye), r, facecolor="white", edgecolor=LINE_DARK, lw=0.9, zorder=5))
    neck, hip = eye - r, eye - 0.27
    kw = dict(color=LINE_DARK, lw=0.9, zorder=5)
    ax.plot([x, x], [neck, hip], **kw)
    ax.plot([x - 0.09, x, x + 0.09], [g, hip, g], **kw)
    ax.plot([x - 0.11, x, x + 0.11], [hip + 0.03, neck - 0.07, hip + 0.03], **kw)


def _shelter(ax, x0=4.2):
    g, bh, eye_h = 2.4, 0.9, 0.5
    ax.plot([x0, x0 + 2.05], [g, g], color=LINE_DARK, lw=0.9, zorder=2)
    bx1 = x0 + 0.55
    _rect(ax, x0, g, bx1, g + bh)
    px, eye = x0 + 1.25, g + eye_h
    _person(ax, px, g, eye)
    top = (bx1, g + bh)
    ax.plot([px, top[0]], [eye, top[1]], color=WALL_FACING, lw=1.3, zorder=4)
    ax.plot([bx1 + 0.02, px], [eye, eye], color=EDGE, lw=0.7, ls=(0, (2, 2)), zorder=4)
    ang = math.degrees(math.atan2(top[1] - eye, px - top[0]))
    r = 0.45
    t = np.radians(np.linspace(180 - ang, 180, 40))
    ax.plot(px + r * np.cos(t), eye + r * np.sin(t), color=WALL_FACING, lw=1.3, zorder=4)
    _txt(ax, px - 0.25, g + 0.9, "upwind\nshelter angle", linespacing=1.0)
    _txt(ax, px + 0.14, eye, "eye 1.5 m")
    _arrow(ax, (x0, _ROW_TOP - 0.02), (x0 + 0.4, _ROW_TOP - 0.02))
    _txt(ax, x0 + 0.5, _ROW_TOP - 0.02, "wind")
    _txt(ax, x0, _TOP_BASE + 0.05, "searched up to 100 m upwind")


def _regimes(ax, route_median_ratio):
    g, w = 0.85, 1.9
    xs0 = (0.08, 2.2, 4.3)
    _txt(ax, 0.05, 1.78, "When the wind blows across the street", fontweight="bold")
    specs = (
        dict(bw=0.3, h=0.4, street=1.3, name="Isolated roughness flow",
             rule=f"height-to-width ratio\nbelow about {RATIO_ISOLATED_MAX:.2f}"),
        dict(bw=0.55, h=0.5, street=0.8, name="Wake interference flow",
             rule=f"height-to-width ratio\nabout {RATIO_ISOLATED_MAX:.2f} to {RATIO_SKIMMING_MIN:.2f}"),
        dict(bw=0.7, h=0.6, street=0.5, name="Skimming flow",
             rule=f"height-to-width ratio\nabove about {RATIO_SKIMMING_MIN:.2f}"),
    )
    for i, (x0, s) in enumerate(zip(xs0, specs)):
        bw, h, st = s["bw"], s["h"], s["street"]
        lx1, rx0, rx1 = x0 + bw, x0 + bw + st, x0 + w
        ax.plot([x0, rx1], [g, g], color=LINE_DARK, lw=0.9, zorder=2)
        _rect(ax, x0, g, lx1, g + h)
        _rect(ax, rx0, g, rx1, g + h)
        top = g + h
        mid = (lx1 + rx0) / 2
        if i == 0:
            _stream(ax, [x0, lx1, lx1 + 0.3, mid, rx0 - 0.3, rx0, rx1],
                    [top + 0.2, top + 0.1, g + 0.2, g + 0.1, top + 0.03, top + 0.1, top + 0.2])
            _stream(ax, [x0, rx1], [top + 0.28, top + 0.28])
        elif i == 1:
            _stream(ax, [x0, lx1, lx1 + 0.2, mid, rx0 - 0.25, rx0 - 0.12, rx0 + 0.2, rx1],
                    [top + 0.2, top + 0.1, top - 0.1, g + 0.2, g + 0.3, top + 0.04, top + 0.16, top + 0.22])
            _stream(ax, [x0, rx1], [top + 0.28, top + 0.28])
            _loop(ax, lx1 + 0.17, g + 0.18, 0.1, 0.1, a0=160, a1=-120)
        else:
            _stream(ax, [x0, lx1, rx0, rx1], [top + 0.2, top + 0.15, top + 0.15, top + 0.2])
            _loop(ax, mid, g + h / 2 + 0.02, 0.17, 0.2, a0=105, a1=-210)
        cx = x0 + w / 2
        _txt(ax, x0, 0.63, s["name"], fontweight="bold")
        _txt(ax, x0, 0.49, s["rule"], va="top", linespacing=1.1)
        if i == 2 and route_median_ratio is not None:
            _txt(ax, x0, 0.1, f"this route: median ratio {route_median_ratio:.1f}", color=WALL_EDGE,
                 fontweight="bold")


def _draw(ax, route_median_ratio=None):
    ax.set_xlim(0, TEXT_WIDTH_IN)
    ax.set_ylim(0, FIG_HEIGHT_IN)
    ax.set_axis_off()
    _header(ax, 0.05, "a", "Frontal area density")
    _header(ax, 1.95, "b", "Canyon alignment")
    _header(ax, 4.2, "c", "Upwind shelter angle")
    _frontal(ax)
    _alignment(ax)
    _shelter(ax)
    ax.plot([0.05, TEXT_WIDTH_IN - 0.05], [1.99, 1.99], color="#dddddd", lw=0.6, zorder=1)
    _regimes(ax, route_median_ratio)


def build_fig_vent_schematic(out_path: Path, route_median_ratio: float | None = None) -> Path:
    import matplotlib.pyplot as plt

    with figure_style():
        fig = plt.figure(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        _draw(ax, route_median_ratio)
        return save(fig, Path(out_path))

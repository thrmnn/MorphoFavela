"""Shared look of the OM2 report figures.

Every figure is drawn at its print width (A4, 2.5 cm margins: 16 cm = 6.3 in)
with 8 to 9 pt text, so nothing is scaled when the report is typeset.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import matplotlib
import matplotlib.patheffects as mpe
import matplotlib.ticker
import numpy as np
import pandas as pd

#: Same family name the report CSS uses.
FONT_FAMILY = "DejaVu Sans"
TEXT_WIDTH_IN = 6.3
FONT_PT = 8.0
FONT_PT_SMALL = 8.0
DPI = 200

#: Sun scale: dark blue (little sun) to yellow (much sun), lightness rising
#: monotonically (CIELAB L 26 to 91) with no grey middle, so "lighter = more
#: sun" reads at a glance.
SUN_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "om_sun", ["#183c7c", "#2a68ad", "#4c97c8", "#93c5d2", "#d3e09a", "#fde74c"])
if "om_sun" not in matplotlib.colormaps:
    matplotlib.colormaps.register(SUN_CMAP)

#: One colour scale per variable, reused wherever the variable is drawn as a colour.
VAR_CMAP = {
    "sun_share": "om_sun",
    "sun_dose": "om_sun",
    "shelter_angle": "magma_r",
}
#: No direct sun at all: darker than the darkest scale colour (L 6 against 26).
NO_SUN = "#0a1128"

LINE_DARK = "#222222"
POINT_GREY = "#b8b8b8"
#: Sensor-matched lines: not the regime colours (orange and blue).
TAU_COLOURS = {10: "#009E73", 30: "#CC79A7"}
#: Soft tints for the neighbourhood band.
BAND_TINTS = ["#d9d9d9", "#bfd3c1", "#e6d3a8", "#c9c3dc"]

_BUILDING_FACE = "#e6e6e6"
_BUILDING_EDGE = "#a6a6a6"
_HALO = [mpe.withStroke(linewidth=2.4, foreground="white")]


@contextmanager
def figure_style():
    """rcParams for one figure, restored on exit."""
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
    }
    with matplotlib.rc_context(rc):
        yield


def save(fig, path):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Print size is the contract (text width, fonts >= 8 pt): never inherit a
    # global savefig.bbox="tight" that would crop the canvas.
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return Path(path)


def ten_m_means(frame: pd.DataFrame, columns, step_m: float = 10.0) -> pd.DataFrame:
    """Means of 1 m values in step_m bins of distance; x = bin centre."""
    d = frame["distance_along_m"].to_numpy(float)
    b = np.floor(d / step_m).astype(int)
    out = frame[list(columns)].groupby(b).mean()
    out.insert(0, "x", (out.index.to_numpy() + 0.5) * step_m)
    return out.reset_index(drop=True)


def neighbourhood_stretches(points: pd.DataFrame) -> list[dict]:
    """Runs of one neighbourhood along the route. Points with no name take the
    name of the stretch they sit in (nearest named point along the route)."""
    p = points.sort_values("distance_along_m")
    name = p["neighbourhood"].copy()
    name = name.ffill().bfill()
    run = (name != name.shift()).cumsum()
    out = []
    for _, g in p.assign(_n=name).groupby(run.to_numpy()):
        out.append({"name": str(g["_n"].iloc[0]), "start": float(g["distance_along_m"].min()),
                    "end": float(g["distance_along_m"].max())})
    for a, b in zip(out[:-1], out[1:]):
        mid = (a["end"] + b["start"]) / 2
        a["end"] = b["start"] = mid
    out[0]["start"] = 0.0
    return out


def _wrap_name(name: str, width_m: float, total_m: float, axes_in: float) -> str:
    """Two lines when the stretch is too narrow for one at 8 pt."""
    needed_in = 0.062 * len(name)
    if needed_in > axes_in * width_m / total_m and " " in name:
        head, _, tail = name.partition(" ")
        return f"{head}\n{tail}"
    return name


def draw_neighbourhood_band(ax, stretches: list[dict], total_m: float, axes_in: float = 5.2):
    """Coloured spans with the stretch names, in a thin axes above a profile.
    Shared by the form and ventilation profile figures."""
    names = []
    for s in stretches:
        if s["name"] not in names:
            names.append(s["name"])
    for s in stretches:
        colour = BAND_TINTS[names.index(s["name"]) % len(BAND_TINTS)]
        ax.axvspan(s["start"], s["end"], ymin=0.0, ymax=0.32, color=colour, lw=0)
        label = _wrap_name(s["name"], s["end"] - s["start"], total_m, axes_in)
        ax.text((s["start"] + s["end"]) / 2, 0.40, label, ha="center", va="bottom", fontsize=FONT_PT,
                linespacing=0.95)
    ax.set_xlim(0, total_m)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def distance_axis(ax, total_m: float, step: float = 250.0):
    ax.set_xlim(0, total_m)
    ax.set_xticks(np.arange(0, total_m + 1, step))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def route_extent(points: pd.DataFrame, margin_m: float = 30.0):
    return (float(points["x"].min()) - margin_m, float(points["x"].max()) + margin_m,
            float(points["y"].min()) - margin_m, float(points["y"].max()) + margin_m)


def draw_buildings(ax, buildings, extent):
    xmin, xmax, ymin, ymax = extent
    if buildings is not None and len(buildings):
        clipped = buildings.cx[xmin:xmax, ymin:ymax]
        if len(clipped):
            clipped.plot(ax=ax, facecolor=_BUILDING_FACE, edgecolor=_BUILDING_EDGE, linewidth=0.25, zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def map_height_in(extent, width_in: float) -> float:
    xmin, xmax, ymin, ymax = extent
    return width_in * (ymax - ymin) / (xmax - xmin)


def fit_map_width_in(extent, max_height_in: float, max_width_in: float = TEXT_WIDTH_IN) -> float:
    """Map width at which the equal-aspect extent is no taller than max_height_in."""
    xmin, xmax, ymin, ymax = extent
    return min(max_width_in, max_height_in * (xmax - xmin) / (ymax - ymin))


def north_arrow(ax, loc=(0.92, 0.90), size=0.09):
    ax.annotate("", xy=(loc[0], loc[1]), xytext=(loc[0], loc[1] - size), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10), zorder=9)
    ax.text(loc[0], loc[1] + 0.01, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=FONT_PT,
            fontweight="bold", zorder=9, path_effects=_HALO)


def scale_bar(ax, length_m: float = 100.0, loc=(0.04, 0.04)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = x0 + (x1 - x0) * loc[0]
    sy = y0 + (y1 - y0) * loc[1]
    ax.plot([sx, sx + length_m], [sy, sy], color="black", lw=2.2, solid_capstyle="butt", zorder=9)
    ax.text(sx + length_m / 2, sy + (y1 - y0) * 0.012, f"{length_m:.0f} m", ha="center", va="bottom",
            fontsize=FONT_PT, zorder=9, path_effects=_HALO)


def wind_arrow(ax, wind_from_deg: float, colour: str, label: str, centre=(0.17, 0.84), half=0.08):
    """Arrow in axes coordinates pointing where the wind blows to."""
    to = math.radians((wind_from_deg + 180.0) % 360.0)
    dx, dy = math.sin(to), math.cos(to)
    cx, cy = centre
    ax.annotate("", xy=(cx + half * dx, cy + half * dy), xytext=(cx - half * dx, cy - half * dy),
                xycoords="axes fraction", textcoords="axes fraction", zorder=9,
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=colour, mutation_scale=16))
    ax.text(cx, cy - half - 0.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=FONT_PT,
            color="black", zorder=9, path_effects=_HALO)


HALO = _HALO


def place_labels(fig, ax, items, avoid_xy, radii=(7, 12, 18, 26, 36, 48), n_angles=16):
    """Place text next to anchor points without overlapping each other or the
    route. items: dicts with xy (data coords), text and optional kw. Candidates
    are tried nearest first; the least-overlapping one wins."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid = ax.transData.transform(np.asarray(avoid_xy, float))
    placed = []
    angles = [i * 2 * math.pi / n_angles for i in range(n_angles)]
    for it in items:
        best = None
        for r in radii:
            for a in angles:
                dx, dy = math.cos(a), math.sin(a)
                ha = "left" if dx > 0.35 else "right" if dx < -0.35 else "center"
                va = "bottom" if dy > 0.35 else "top" if dy < -0.35 else "center"
                t = ax.annotate(it["text"], it["xy"], xytext=(r * dx, r * dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=FONT_PT, zorder=8, path_effects=_HALO, **it.get("kw", {}))
                bb = t.get_window_extent(renderer).expanded(1.12, 1.25)
                hits = sum(bb.overlaps(q) for q in placed)
                inside = (avoid[:, 0] > bb.x0) & (avoid[:, 0] < bb.x1) & (avoid[:, 1] > bb.y0) & (avoid[:, 1] < bb.y1)
                score = hits * 1000 + int(inside.sum()) + r * 0.01
                ax_bb = ax.get_window_extent(renderer)
                if bb.x0 < ax_bb.x0 or bb.x1 > ax_bb.x1 or bb.y0 < ax_bb.y0 or bb.y1 > ax_bb.y1:
                    score += 5000
                if best is None or score < best[0]:
                    if best is not None:
                        best[2].remove()
                    best = (score, bb, t)
                else:
                    t.remove()
                if best[0] < 1:
                    break
            if best[0] < 1:
                break
        placed.append(best[1])


FLAG_LABEL = "route inside a building outline or more than 10 m from a mapped street"
FLAG_MIN_M = 5.0


def flagged_spans(points: pd.DataFrame, min_len_m: float = FLAG_MIN_M) -> list[tuple[float, float]]:
    """Distance spans of route_geometry_flag runs: runs closer than min_len_m are
    merged, runs shorter than min_len_m are dropped."""
    if "route_geometry_flag" not in points.columns:
        return []
    p = points.sort_values("distance_along_m")
    f = p["route_geometry_flag"].astype(bool).to_numpy()
    d = p["distance_along_m"].to_numpy(float)
    runs, st = [], None
    for i, v in enumerate(f):
        if v and st is None:
            st = i
        if not v and st is not None:
            runs.append([d[st], d[i - 1]])
            st = None
    if st is not None:
        runs.append([d[st], d[-1]])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_len_m:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a + 1 >= min_len_m]


def shade_flagged(axes, spans):
    for ax in axes:
        for a, b in spans:
            ax.axvspan(a, b + 1, facecolor="#ececec", edgecolor="#b5b5b5", hatch="////", lw=0, zorder=0)


def flag_handle():
    from matplotlib.patches import Patch

    return Patch(facecolor="#ececec", edgecolor="#b5b5b5", hatch="////", lw=0, label=FLAG_LABEL)


def flag_facts(spans) -> dict:
    return {"flagged_spans_m": [[float(a), float(b)] for a, b in spans],
            "flagged_total_length_m": float(sum(b - a + 1 for a, b in spans)),
            "flagged_min_run_m": FLAG_MIN_M}

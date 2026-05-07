"""Figure 6 — SVF↔solar dissociation on Vidigal's south face.

SVF on x, hours of direct winter sun on y. Underlay is a low-alpha
scatter of every street-level sample, coloured by terrain aspect
using a cyclic palette (N=blue, E=green, S=red, W=orange). Overlaid
are four median traces — one per terrain-aspect quadrant — with
shaded ±IQR bands. Trace segments where the SVF bin contains fewer
than 30 cells are dashed (open markers) to make the small-sample
tail (W-quadrant beyond SVF≈0.65) visually fragile rather than
visually authoritative.

Vidigal-only because the ground-solar pipeline has run only there;
SVF is universally available. The spatial expression of the same
dissociation is in `figS_terrain_aspect_spatial.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    PROJECT_ROOT,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)
sys.path.insert(0, str(PROJECT_ROOT))
from src.morphometry.aspect import aspect_quadrant  # noqa: E402

# Compass-meaningful palette for quadrant traces; underlay scatter uses
# a smooth cyclic interpolation of these endpoints.
QUADRANT_COLORS = {
    "N": "#1F6FB0",  # cool blue
    "E": "#2CA02C",  # green
    "S": "#D62728",  # warm red
    "W": "#FF7F0E",  # orange
}
QUADRANTS = ("N", "E", "S", "W")
SLOPE_THRESHOLD = 5.0
MIN_BIN_N = 30


def _build_compass_cmap() -> ListedColormap:
    """Cyclic colormap: N(0°) blue → E(90°) green → S(180°) red → W(270°) orange → N."""
    from matplotlib.colors import to_rgb

    anchors = [
        (0.0, QUADRANT_COLORS["N"]),
        (0.25, QUADRANT_COLORS["E"]),
        (0.50, QUADRANT_COLORS["S"]),
        (0.75, QUADRANT_COLORS["W"]),
        (1.0, QUADRANT_COLORS["N"]),
    ]
    n_steps = 256
    rgb = np.zeros((n_steps, 3))
    for i in range(n_steps):
        t = i / (n_steps - 1)
        # find segment
        for j in range(len(anchors) - 1):
            t0, c0 = anchors[j]
            t1, c1 = anchors[j + 1]
            if t0 <= t <= t1:
                local = (t - t0) / (t1 - t0)
                rgb0 = np.array(to_rgb(c0))
                rgb1 = np.array(to_rgb(c1))
                rgb[i] = rgb0 * (1 - local) + rgb1 * local
                break
    return ListedColormap(rgb, name="compass")


def _load_data() -> gpd.GeoDataFrame:
    solar_path = (
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    grid_path = (
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    solar = gpd.read_file(solar_path)
    grid = gpd.read_file(grid_path)
    if solar.crs != grid.crs:
        solar = solar.to_crs(grid.crs)
    keep = grid[["geometry", "aspect_deg", "slope_deg"]]
    pts = gpd.sjoin(solar, keep, how="left", predicate="within").drop(columns="index_right")
    pts = pts.dropna(subset=["aspect_deg", "svf", "solar_hours"])
    pts = pts[pts["slope_deg"].fillna(0) >= SLOPE_THRESHOLD].copy()
    pts["quadrant"] = aspect_quadrant(pts["aspect_deg"].values)
    return pts


def _quadrant_trace(
    sub: gpd.GeoDataFrame,
    svf_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin median + IQR + count for a quadrant subset."""
    centres = 0.5 * (svf_edges[:-1] + svf_edges[1:])
    idx = np.digitize(sub["svf"].values, svf_edges) - 1
    idx = np.clip(idx, 0, len(centres) - 1)
    med = np.full(len(centres), np.nan)
    q25 = np.full(len(centres), np.nan)
    q75 = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        vals = sub["solar_hours"].values[idx == i]
        counts[i] = len(vals)
        if len(vals) > 0:
            med[i] = np.median(vals)
            q25[i] = np.percentile(vals, 25)
            q75[i] = np.percentile(vals, 75)
    return centres, med, q25, q75, counts


def _draw_main_panel(ax: plt.Axes, pts: gpd.GeoDataFrame) -> None:
    cmap = _build_compass_cmap()

    # Underlay: aspect-colored scatter, very faint
    ax.scatter(
        pts["svf"],
        pts["solar_hours"],
        c=pts["aspect_deg"] / 360.0,
        cmap=cmap,
        s=2.5,
        alpha=0.18,
        linewidths=0,
        zorder=1,
    )

    # Per-quadrant traces with IQR bands
    svf_edges = np.linspace(0, 1, 11)
    for q in QUADRANTS:
        sub = pts[pts["quadrant"] == q]
        if sub.empty:
            continue
        centres, med, q25, q75, counts = _quadrant_trace(sub, svf_edges)
        finite = np.isfinite(med)
        # IQR band (only where bin has solid sample)
        solid = finite & (counts >= MIN_BIN_N)
        if solid.any():
            ax.fill_between(
                centres[solid],
                q25[solid],
                q75[solid],
                color=QUADRANT_COLORS[q],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        # Solid line where n>=MIN_BIN_N, dashed where smaller
        if solid.any():
            ax.plot(
                centres[solid],
                med[solid],
                color=QUADRANT_COLORS[q],
                lw=1.8,
                marker="o",
                ms=4,
                zorder=3,
            )
        weak = finite & (counts < MIN_BIN_N) & (counts > 0)
        if weak.any():
            ax.plot(
                centres[weak],
                med[weak],
                color=QUADRANT_COLORS[q],
                lw=1.0,
                ls=(0, (2, 1.5)),
                marker="o",
                ms=3,
                mfc="white",
                mew=0.8,
                alpha=0.7,
                zorder=3,
            )

    # Headline annotation: gap between N and S at SVF~0.5
    ax.annotate(
        "",
        xy=(0.50, 5.0),
        xytext=(0.50, 0.0),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.8),
        zorder=4,
    )
    ax.text(
        0.515,
        2.5,
        "5 hr gap\nat same SVF",
        fontsize=6.5,
        ha="left",
        va="center",
        color="0.15",
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 10.5)
    ax.set_xlabel("SVF")
    ax.set_ylabel("hours of direct sun (winter solstice)")
    ax.axhline(0, color="0.85", lw=0.4, zorder=0)


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=QUADRANT_COLORS["N"], lw=1.8, marker="o", ms=4, label="N-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["E"], lw=1.8, marker="o", ms=4, label="E-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["S"], lw=1.8, marker="o", ms=4, label="S-facing"),
        Line2D([0], [0], color=QUADRANT_COLORS["W"], lw=1.8, marker="o", ms=4, label="W-facing"),
        Patch(facecolor="0.7", edgecolor="none", alpha=0.4, label="±IQR (n≥30)"),
        Line2D([0], [0], color="0.4", lw=1.0, ls=(0, (2, 1.5)), marker="o", ms=3, mfc="white",
               label="n<30 (sparse)"),
    ]
    ax.legend(
        handles=handles,
        title="terrain aspect quadrant",
        title_fontsize=6,
        fontsize=5.5,
        loc="upper left",
        frameon=False,
        ncol=1,
        handlelength=1.5,
    )


def main() -> None:
    apply_style()
    pts = _load_data()

    fig = plt.figure(figsize=(WIDTH_DOUBLE * 0.66, WIDTH_DOUBLE * 0.50))
    ax_main = fig.add_axes([0.10, 0.12, 0.85, 0.78])
    _draw_main_panel(ax_main, pts)
    _draw_legend(ax_main)

    fig.suptitle(
        "On Vidigal's south face, terrain aspect adds 4–5 hr of solar variation at any SVF\n"
        "winter sun in the southern hemisphere bypasses S-facing slopes",
        fontsize=8.5,
        y=0.97,
    )
    ax_main.text(
        0.99,
        0.02,
        f"n = {len(pts):,}  •  Vidigal street points, slope ≥ {int(SLOPE_THRESHOLD)}°",
        transform=ax_main.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="0.4",
    )

    save_fig(fig, "fig06_terrain_aspect")


if __name__ == "__main__":
    main()

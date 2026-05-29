"""Figure S — spatial expression of the SVF↔solar dissociation on Vidigal.

Four panels, all on the same Vidigal street-point dataset:
  (a) terrain aspect, compass-coloured (N=blue, E=green, S=red, W=orange)
  (b) SVF                                       (viridis 0–1)
  (c) winter-solstice solar hours (normalised)   (viridis 0–1)
  (d) DISAGREEMENT  (SVF_norm − solar_norm)      (RdBu, ±0.6)

Panel (d) is the headline: red zones are where the sky is open but the
sun does not reach (high SVF, low solar — typically S-facing cells in
the southern-hemisphere winter); blue zones are the opposite.

The aspect palette is shared with Figure 6's quadrant traces so the two
figures cross-reference visually: an S-facing red splash on (a) maps to
the S-facing red trace in Figure 6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    PROJECT_ROOT,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    load_buildings,
    save_fig,
)

QUADRANT_COLORS = {
    "N": "#1F6FB0",
    "E": "#2CA02C",
    "S": "#D62728",
    "W": "#FF7F0E",
}


def _build_compass_cmap() -> ListedColormap:
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
        for j in range(len(anchors) - 1):
            t0, c0 = anchors[j]
            t1, c1 = anchors[j + 1]
            if t0 <= t <= t1:
                local = (t - t0) / (t1 - t0)
                rgb[i] = np.array(to_rgb(c0)) * (1 - local) + np.array(to_rgb(c1)) * local
                break
    return ListedColormap(rgb, name="compass")


def main() -> None:
    apply_style()

    solar = gpd.read_file(
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    )
    grid = gpd.read_file(
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid" / "grid_metrics.gpkg"
    )
    if solar.crs != grid.crs:
        solar = solar.to_crs(grid.crs)
    pts = gpd.sjoin(solar, grid[["geometry", "aspect_deg"]], how="left", predicate="within").drop(
        columns="index_right"
    )
    pts = pts.dropna(subset=["aspect_deg", "svf", "solar_hours"])
    pts["solar_norm"] = pts["solar_hours"] / 11.0
    pts["disagreement"] = pts["svf"] - pts["solar_norm"]

    buildings = load_buildings("vidigal", extended=False)
    if buildings.crs != pts.crs:
        buildings = buildings.to_crs(pts.crs)
    bbox = pts.total_bounds
    pad = 30
    xmin, ymin, xmax, ymax = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(WIDTH_DOUBLE * 0.85, WIDTH_DOUBLE * 0.78),
        gridspec_kw={"hspace": 0.18, "wspace": 0.05},
    )
    axes_flat = axes.ravel()
    compass_cmap = _build_compass_cmap()

    panel_specs = [
        ("(a) terrain aspect", pts["aspect_deg"], compass_cmap, 0, 360, "deg from N"),
        ("(b) SVF", pts["svf"], "viridis", 0, 1, "SVF (0–1)"),
        (
            "(c) winter solar hours (normalised)",
            pts["solar_norm"],
            "viridis",
            0,
            1,
            "solar / 11 hr",
        ),
        (
            "(d) disagreement = SVF − solar  (the headline)",
            pts["disagreement"],
            "RdBu_r",
            -0.6,
            0.6,
            "SVF − solar (red: open sky, no sun)",
        ),
    ]

    for ax, (title, values, cmap, vmin, vmax, label) in zip(axes_flat, panel_specs):
        buildings.plot(ax=ax, facecolor="0.86", edgecolor="0.62", linewidth=0.18)
        sc = ax.scatter(
            pts.geometry.x,
            pts.geometry.y,
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=4,
            alpha=0.95,
            linewidths=0,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        clean_map_axes(ax)
        ax.set_title(title, fontsize=8, loc="left", pad=2)
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, aspect=22)
        cbar.set_label(label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        if title.startswith("(a)"):
            cbar.set_ticks([0, 90, 180, 270, 360])
            cbar.set_ticklabels(["N", "E", "S", "W", "N"])
        if title.startswith("(d)"):
            cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

    # Pull the eye to the disagreement panel
    add_scalebar(axes_flat[3], length_m=200)
    add_north_arrow(axes_flat[3])

    fig.suptitle(
        "Vidigal — same point set, four encodings.  Panel (d) shows where SVF and\n"
        "winter solar hours disagree most: red = open sky but no sun (S-facing slopes).",
        fontsize=9,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "figS_terrain_aspect_spatial")


if __name__ == "__main__":
    main()

"""Figure 6 — Terrain-aspect decomposition across hillside favelas.

Northness (cos(aspect)) and eastness (sin(aspect)) component maps for the
three hillside sites (Vidigal, Rocinha, Complexo do Alemão) at 10 m grid
resolution. The circular (sin, cos) decomposition handles the 0/360°
discontinuity of raw aspect and lets near-flat cells register near zero on
both axes.

Layout: 2 rows (northness, eastness) × 3 columns (the three hillside sites).
Each cell is a small spatial point map of the 10 m grid centroids coloured by
the component value on a diverging RdBu colormap (−1..+1), one shared colorbar
per row.

Aspect convention (src/morphometry/aspect.py): bearing of the downslope
direction, 0=N, 90=E, 180=S, 270=W. northness = cos(aspect_rad),
eastness = sin(aspect_rad).

Inputs:
  outputs/{site}/morphometrics/grid/grid_metrics.gpkg  (aspect_deg, centroid_x/y)

Output:
  outputs/paper_figures/exports/fig06_terrain_aspect.png (+ .svg)
  copied downstream to artifacts/latex/figures/ by the caller.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (
    PROJECT_ROOT,
    SITE_LABELS,
    WIDTH_DOUBLE,
    apply_style,
    save_fig,
)

HILLSIDE_SITES = ["vidigal", "rocinha", "complexo_do_alemao"]

CMAP = "RdBu"          # diverging; red ↔ south/west, blue ↔ north/east
VLIM = 1.0
INK = "#1F2933"

COMPONENTS = [
    ("northness", "Northness  (cos aspect)", "S-facing  ←   →  N-facing"),
    ("eastness", "Eastness  (sin aspect)", "W-facing  ←   →  E-facing"),
]


def _load_components(site: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, northness, eastness) arrays for a site's grid centroids."""
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Grid not found for {site}: {path}")
    g = gpd.read_file(path)
    if "aspect_deg" not in g.columns:
        raise KeyError(f"{site}: grid_metrics.gpkg has no aspect_deg column")

    if {"centroid_x", "centroid_y"}.issubset(g.columns):
        x = g["centroid_x"].to_numpy()
        y = g["centroid_y"].to_numpy()
    else:
        cent = g.geometry.centroid
        x = cent.x.to_numpy()
        y = cent.y.to_numpy()

    rad = np.deg2rad(g["aspect_deg"].to_numpy())
    northness = np.cos(rad)
    eastness = np.sin(rad)
    return x, y, northness, eastness


def _draw_map(ax: plt.Axes, x: np.ndarray, y: np.ndarray, vals: np.ndarray):
    sc = ax.scatter(
        x,
        y,
        c=vals,
        cmap=CMAP,
        vmin=-VLIM,
        vmax=VLIM,
        s=2.6,
        marker="s",
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc


def main() -> None:
    apply_style()

    print("Loading hillside grids ...")
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for site in HILLSIDE_SITES:
        x, y, n, e = _load_components(site)
        data[site] = (x, y, n, e)
        print(f"  {site:<20} cells={len(x):6,}  "
              f"northness[{n.min():+.2f},{n.max():+.2f}]  "
              f"eastness[{e.min():+.2f},{e.max():+.2f}]")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.74))
    # 2 rows of maps + a thin colorbar column on the right.
    outer = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.06],
        hspace=0.10,
        wspace=0.06,
        left=0.045,
        right=0.93,
        top=0.86,
        bottom=0.04,
    )

    for row, (comp, comp_label, comp_scale) in enumerate(COMPONENTS):
        sc = None
        for col, site in enumerate(HILLSIDE_SITES):
            ax = fig.add_subplot(outer[row, col])
            x, y, n, e = data[site]
            vals = n if comp == "northness" else e
            sc = _draw_map(ax, x, y, vals)
            if row == 0:
                ax.set_title(SITE_LABELS[site], fontsize=8.5, fontweight="bold",
                             color=INK, pad=4)
            if col == 0:
                ax.text(
                    -0.04,
                    0.5,
                    comp_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=INK,
                )

        cax = fig.add_subplot(outer[row, 3])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb.ax.tick_params(labelsize=6, length=2, width=0.4)
        cb.set_label(comp_scale, fontsize=6.2, labelpad=3)
        cb.outline.set_linewidth(0.4)

    fig.suptitle(
        "Terrain-aspect decomposition across hillside favelas",
        fontsize=10,
        y=0.965,
        fontweight="bold",
        color="#1a1a1a",
    )
    fig.text(
        0.5,
        0.905,
        "Northness (cos aspect) and eastness (sin aspect) component maps of the "
        "10 m morphometric grid. The circular decomposition handles the 0/360° "
        "discontinuity; near-flat cells register near zero on both axes.",
        ha="center",
        va="top",
        fontsize=6.0,
        style="italic",
        color="#666",
    )

    save_fig(fig, "fig06_terrain_aspect")


if __name__ == "__main__":
    main()

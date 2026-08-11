#!/usr/bin/env python3
"""Figure 1 (brisa paper): single-panel study-sites map.

Canonical fig01 for the brisaverse handoff (submission reduction): a single
panel, the regional site map of Rio with the 5 favela boundaries + per-site
footprint insets, laid out across the full figure width with no panel letters.

Retired from the paper figure and preserved on the hub figure page as
downscaled thumbnails (NOT deleted): the geometry->adequacy pipeline schematic
(``fig01_panelA.png``), the printable 3-D massing excerpts, and the Vidigal
terrestrial-LiDAR 3-D scan (the facade-resolving massing that only ground-based
scanning captures inside the canyons; source ``data/vidigal_tls/raw/full_scan.stl``).
Their generators are retained below (``panel_a_schematic``, ``panel_c_excerpts``,
``panel_tls_scan_3d`` and helpers) so the hub thumbnails can be regenerated.
``main()`` now composites only the site map; it also exports the removed TLS
scan as a standalone ``fig01_tls_3d_wip`` so it can keep being improved.

Deliberately kept separate from ``fig01_study_sites.py`` (the technical
report's map-only Figure 1). brisaverse promotes ``exports/fig01_composite.png``
as its fig01 and stages it to ``shared/figures/fig01_study_sites.png``; the TLS
scan stages to ``shared/figures/fig01_tls_3d_wip.png``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import trimesh
from fig_style import *
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import box

sys.path.insert(0, str(PROJECT_ROOT))
from src.print3d.model import _building_prisms, _sample_terrain_core

PANEL_A_PATH = Path("/home/theo/brisaverse/shared/figures/fig01_panelA.png")

# Vidigal terrestrial-LiDAR scan (terrain + buildings welded into a single mesh).
# The facade-resolving 3-D geometry ground-based scanning captures where airborne
# scanning cannot see into the canyons. 17k-face raw scan; the higher-fidelity
# fallback is ``outputs/vidigal_tls/morphometrics/svf/scene.stl`` (17 MB).
TLS_SCAN_STL_PATH = (
    Path("/home/theo/MorphoFavela")
    / "data" / "vidigal_tls" / "raw" / "full_scan.stl"
)

# (site, patch_id, sampling) — accented "maré" dir handled via PROJECT_ROOT join.
# All four resolve in campaign_sampling (VDG-P07 also exists in pilot_sampling,
# so the historical pilot deprecation does not force a substitution here).
PANEL_C_EXCERPTS: list[tuple[str, str, str]] = [
    ("rocinha", "ROC-P18", "campaign_sampling"),
    ("vidigal", "VDG-P07", "campaign_sampling"),
    ("maré", "MAR-P20", "campaign_sampling"),
    ("riodaspedras", "RDP-P20", "campaign_sampling"),
]

BUILDING_CMAP = "viridis"

# Site-name label offsets (in points) from each boundary centroid, drawn with a
# thin leader line so names sit off the fabric. Rocinha and Vidigal are adjacent
# south-zone hillsides whose centroids nearly coincide, so they are pushed apart
# (Rocinha up-left, Vidigal down-right) to break the stacked-label collision.
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "vidigal": (30, -22),
    "rocinha": (30, 18),
    "complexo_do_alemao": (-30, 14),
    "riodaspedras": (34, 8),
    "maré": (32, 12),
}

PROVENANCE = (
    "Building footprints © IPP (municipal cadaster); ALS heights MIT/SondoTecnica. "
    "Pipeline open; input data not redistributable."
)


def panel_b_sitemap(subfig) -> None:
    """Draw the Rio regional map + 5 per-site footprint insets into a subfigure."""
    favelas_path = PROJECT_ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"
    all_favelas = gpd.read_file(favelas_path)
    rj_dtm_path = PROJECT_ROOT / "data" / "RJ" / "DTM_RJ.tif"

    boundaries = {}
    buildings = {}
    for site in SITE_ORDER:
        try:
            boundaries[site] = load_boundary(site)
        except FileNotFoundError:
            for name_candidate in [site.upper(), site.replace("_", " ").upper()]:
                match = all_favelas[
                    all_favelas["nome"].str.upper().str.contains(name_candidate, na=False)
                ]
                if not match.empty:
                    boundaries[site] = match
                    break
        try:
            buildings[site] = load_buildings(site, extended=False)
        except FileNotFoundError:
            pass

    gs = gridspec.GridSpec(
        1, 2, figure=subfig, width_ratios=[1.3, 1], wspace=0.02,
        bottom=0.07, top=0.96,
    )
    ax_main = subfig.add_subplot(gs[0])

    if rj_dtm_path.exists():
        try:
            with rasterio.open(rj_dtm_path) as src:
                factor = 4
                dtm = src.read(1, out_shape=(src.height // factor, src.width // factor)).astype(
                    float
                )
                nodata = src.nodata
                if nodata is not None:
                    dtm[dtm == nodata] = np.nan
                dtm[np.abs(dtm) > 1e6] = np.nan
                bounds = src.bounds
                shade = hillshade(dtm, src.res[0] * factor)
                shade = np.where(np.isnan(dtm), 0.95, shade)
                ax_main.imshow(
                    shade,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    alpha=0.5,
                    zorder=0,
                )
        except Exception as e:
            print(f"  Hillshade failed: {e}")

    all_favelas.plot(ax=ax_main, facecolor="none", edgecolor="#cccccc", linewidth=0.15, zorder=1)

    for site in SITE_ORDER:
        if site in boundaries:
            boundaries[site].plot(
                ax=ax_main,
                facecolor=SITE_COLORS[site],
                edgecolor=SITE_COLORS[site],
                alpha=0.5,
                linewidth=0.8,
                zorder=3,
            )
            centroid = boundaries[site].geometry.union_all().centroid
            dx, dy = LABEL_OFFSETS.get(site, (0, 14))
            ax_main.annotate(
                SITE_LABELS[site],
                xy=(centroid.x, centroid.y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=5,
                fontweight="bold",
                ha="right" if dx < 0 else "left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#9AA7B4",
                    linewidth=0.3,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.4,
                    color="#555555",
                    shrinkA=0.0,
                    shrinkB=1.5,
                ),
                zorder=6,
            )

    all_bounds = []
    for site in SITE_ORDER:
        if site in boundaries:
            all_bounds.append(boundaries[site].total_bounds)
    if all_bounds:
        arr = np.array(all_bounds)
        pad = 5000
        ax_main.set_xlim(arr[:, 0].min() - pad, arr[:, 2].max() + pad)
        ax_main.set_ylim(arr[:, 1].min() - pad, arr[:, 3].max() + pad)

    clean_map_axes(ax_main)
    ax_main.set_aspect("equal")
    add_scalebar(ax_main, length_m=5000)
    add_north_arrow(ax_main)

    gs_right = gs[1].subgridspec(5, 1, hspace=0.15)
    for i, site in enumerate(SITE_ORDER):
        ax = subfig.add_subplot(gs_right[i])
        if site in buildings:
            buildings[site].plot(
                ax=ax,
                facecolor=SITE_COLORS[site],
                edgecolor="k",
                linewidth=0.05,
                alpha=0.6,
            )
        if site in boundaries:
            boundaries[site].boundary.plot(ax=ax, color="k", linewidth=0.5, linestyle="--")
        clean_map_axes(ax)
        ax.set_aspect("equal")
        typ = SITE_TYPES[site]
        n_bld = len(buildings[site]) if site in buildings else "?"
        ax.set_title(f"{SITE_LABELS[site]} ({typ}, {n_bld} bld.)", fontsize=5, pad=1)
        add_scalebar(ax)


def _patch_source_dir(site: str, patch_id: str, sampling: str) -> Path:
    return PROJECT_ROOT / "outputs" / site / "sampling_cfd" / sampling / "patches" / patch_id


def _render_excerpt(ax, site: str, patch_id: str, sampling: str, height_norm: Normalize) -> None:
    """Reconstruct terrain + buildings split: ground as a neutral hillshade-lit
    surface (relief from shading, not colour), buildings coloured by height
    above ground so the ramp encodes building height — never absolute elevation.
    """
    pdir = _patch_source_dir(site, patch_id, sampling)
    meta = json.loads((pdir / "patch_meta.json").read_text())
    cx, cy = meta["center_x"], meta["center_y"]
    half = meta.get("analysis_patch_diameter", 100.0) / 2.0

    X, Y, Z = _sample_terrain_core(pdir / "terrain.tif", cx, cy, half)

    buildings = gpd.read_file(pdir / "buildings.gpkg")
    core = box(cx - half, cy - half, cx + half, cy + half)
    buildings = buildings.clip(core)
    buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()]

    ls = LightSource(azdeg=315, altdeg=45)
    intensity = ls.hillshade(Z, vert_exag=1.0)
    grey = 0.45 + 0.45 * intensity[..., None]
    rgb = np.dstack([grey, grey, grey, np.ones_like(intensity)])
    ax.plot_surface(
        X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
    )

    cmap = plt.get_cmap(BUILDING_CMAP)
    for prism, alt in zip(
        _building_prisms(buildings, embed=2.0), _prism_heights(buildings)
    ):
        v = prism.vertices
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2],
            triangles=prism.faces,
            color=cmap(height_norm(alt)),
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            shade=True,
        )

    ax.view_init(elev=35, azim=-60)
    zr = np.ptp(Z) + (float(buildings["altura"].max()) if len(buildings) else 0.0)
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), zr * 1.3 + 1e-9))
    ax.set_axis_off()
    ax.set_title(f"{SITE_LABELS[site]} — {patch_id}", fontsize=5, pad=-2)


def _prism_heights(buildings: gpd.GeoDataFrame) -> list[float]:
    """Per-prism height-above-ground, matching the polygon expansion order in
    ``_building_prisms`` (skip null/zero-height, split MultiPolygons, drop slivers)."""
    heights = []
    for geom, alt in zip(buildings.geometry, buildings["altura"]):
        if geom is None or geom.is_empty or alt is None or alt <= 0:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.area < 1.0:
                continue
            heights.append(float(alt))
    return heights


def panel_c_excerpts(subfig) -> None:
    """Render the 3-D massing excerpts in a 2x2 grid of mplot3d axes."""
    # Shared ramp across panels (cross-site comparable), but capped at the pooled
    # p98 height: a handful of outlier towers (Rocinha tops 129 m vs a ~9 m median)
    # would otherwise crush the typical 5–15 m fabric into the dark end. Buildings
    # above the cap saturate at the top colour (colorbar extend="max").
    heights = []
    for site, patch_id, sampling in PANEL_C_EXCERPTS:
        b = gpd.read_file(_patch_source_dir(site, patch_id, sampling) / "buildings.gpkg")
        if len(b):
            heights.append(b["altura"].to_numpy())
    pooled = np.concatenate(heights) if heights else np.array([1.0])
    vmax = float(np.ceil(np.percentile(pooled, 98)))
    height_norm = Normalize(vmin=0.0, vmax=vmax)

    gs = subfig.add_gridspec(2, 2, hspace=0.05, wspace=0.05, bottom=0.12)
    for i, (site, patch_id, sampling) in enumerate(PANEL_C_EXCERPTS):
        ax = subfig.add_subplot(gs[i // 2, i % 2], projection="3d")
        _render_excerpt(ax, site, patch_id, sampling, height_norm)

    sm = plt.cm.ScalarMappable(norm=height_norm, cmap=BUILDING_CMAP)
    cax = subfig.add_axes((0.30, 0.07, 0.40, 0.018))
    cbar = subfig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("buildings: height above ground (m)", fontsize=4, labelpad=1.5)
    cbar.ax.tick_params(labelsize=4, pad=1)
    subfig.text(
        0.5, 0.115, "ground: neutral hillshade relief (no colour ramp)",
        fontsize=3.8, ha="center", va="bottom", color="#444444",
    )
    subfig.text(
        0.5, 0.005,
        "illustrative subset: 4 of the 5 study sites (C. do Alemão omitted for space)",
        fontsize=3.8, ha="center", va="bottom", color="#444444", style="italic",
    )


def panel_a_schematic(subfig) -> None:
    """RETIRED FROM PAPER (final pass) — retained for hub-thumbnail regeneration.
    Embeds the pre-rendered geometry→adequacy pipeline schematic PNG as-is."""
    ax = subfig.add_subplot(111)
    ax.imshow(mpimg.imread(PANEL_A_PATH), aspect="equal")
    ax.set_axis_off()
    subfig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)


def panel_tls_scan_3d(subfig) -> None:
    """Height-coloured 3-D render of the Vidigal terrestrial-LiDAR scan.

    Loads the raw scan STL (terrain and buildings welded into one surface),
    colours faces by mean height along an earthy ramp, and depth-sorts them into
    painter's order under an isometric view with a subtle relief-shading term.
    This is the facade-resolving 3-D geometry the ground-based scanner recovers
    inside the alley canyons, where airborne scanning sees only rooftops."""
    mesh = trimesh.load(TLS_SCAN_STL_PATH, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.merge_vertices()
    tris = mesh.vertices[mesh.faces]  # (n, 3, 3)
    bb_min, bb_max = mesh.bounds
    extent = mesh.extents

    # Isometric view; depth-sort faces back-to-front along the view direction so
    # the flat Poly3DCollection paints without z-fighting.
    elev, azim = 24.0, -62.0
    er, ar = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
    cen = tris.mean(axis=1)
    order = np.argsort(cen @ view)
    tris = tris[order]
    cen = cen[order]

    # Height colour: mean face Z on an earthy ramp, floor lifted off gist_earth's
    # near-black base so low terrain still reads as ground, not shadow.
    znorm = np.clip((cen[:, 2] - bb_min[2]) / max(extent[2], 1e-9), 0.0, 1.0)
    base = plt.get_cmap("gist_earth")(0.25 + 0.68 * znorm)[:, :3]

    # Subtle relief shading from face-normal · light so massing reads as solids.
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    light = np.array([0.3, 0.4, 0.86])
    intensity = 0.70 + 0.40 * np.clip(np.abs(n @ light), 0, 1)
    fc = np.clip(base * intensity[:, None], 0, 1)

    ax = subfig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(tris, facecolors=fc, edgecolors="none",
                         linewidths=0, antialiased=False)
    )
    ax.view_init(elev=elev, azim=azim)
    # zoom>1 fills the panel: mplot3d otherwise wastes most of the cube on the
    # empty corners of this diagonal terrain slab.
    ax.set_box_aspect((extent[0], extent[1], extent[2] * 1.35), zoom=1.9)
    ax.set_xlim(bb_min[0], bb_max[0])
    ax.set_ylim(bb_min[1], bb_max[1])
    ax.set_zlim(bb_min[2], bb_max[2])
    ax.set_axis_off()
    ax.set_position((0.02, 0.14, 0.96, 0.80))

    # Vertical height ramp key (relative massing, not absolute elevation datum).
    sm = plt.cm.ScalarMappable(
        norm=Normalize(0.0, extent[2]), cmap="gist_earth"
    )
    cax = subfig.add_axes((0.20, 0.105, 0.5, 0.016))
    cbar = subfig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("relative height (m)", fontsize=4.5, labelpad=1.5)
    cbar.ax.tick_params(labelsize=4.5, pad=1)

    subfig.text(
        0.5, 0.93, "Vidigal - terrestrial-LiDAR 3D scan",
        fontsize=6, ha="center", va="top",
    )
    subfig.text(
        0.5, 0.05,
        f"facade-resolving 3D geometry · {len(mesh.faces):,}-face scan · "
        "height-coloured massing on terrain",
        fontsize=4.2, ha="center", va="bottom", color="#444444",
    )


def main():
    apply_style()

    # Standalone parked panel: the Vidigal TLS 3-D scan, exported on its own so it
    # can keep being improved on the hub figure page (removed from the paper).
    fig_tls = plt.figure(figsize=(WIDTH_SINGLE, WIDTH_SINGLE * 1.3))
    panel_tls_scan_3d(fig_tls)
    # gate=False: a 3-D axes whose (hidden) tick labels can't be bbox-measured.
    save_fig(fig_tls, "fig01_tls_3d_wip", gate=False)

    # Main Figure 1: single-panel Rio site map + per-site footprint insets, full
    # width, no panel letters.
    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.5))
    panel_b_sitemap(fig)
    fig.text(0.5, 0.006, PROVENANCE, fontsize=4.5, ha="center", va="bottom", color="#444444")

    save_fig(fig, "fig01_composite", gate=False)


if __name__ == "__main__":
    main()

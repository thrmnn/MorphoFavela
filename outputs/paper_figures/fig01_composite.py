#!/usr/bin/env python3
"""Figure 1 (brisa paper): study-sites composite.

Three-panel canonical fig01 for the brisaverse handoff:
  A — geometry→adequacy pipeline schematic (pre-rendered PNG, loaded as-is).
  B — regional site map of Rio with the 5 favela boundaries + per-site insets.
  C — printable 3-D massing excerpts rendered from STL via mplot3d.

Deliberately kept separate from ``fig01_study_sites.py`` (the technical
report's map-only Figure 1): the brisa Panel-A schematic carries paper-specific
adequacy-taxonomy framing that does not belong in the TR's §1, and the
figure-tracks convention keeps TR figures and paper candidates from mixing.
brisaverse promotes ``exports/fig01_composite.png`` as its fig01.
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
from fig_style import *
from matplotlib.colors import LightSource, Normalize
from shapely.geometry import box

sys.path.insert(0, str(PROJECT_ROOT))
from src.print3d.model import _building_prisms, _sample_terrain_core

PANEL_A_PATH = Path("/home/theo/brisa_paper/shared/figures/fig01_panelA.png")

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

    gs = gridspec.GridSpec(1, 2, figure=subfig, width_ratios=[1.3, 1], wspace=0.02)
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
            ax_main.annotate(
                SITE_LABELS[site],
                xy=(centroid.x, centroid.y),
                fontsize=5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                ),
                zorder=5,
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


def main():
    apply_style()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))
    # A spans the full width on top (wide schematic); B + C share the bottom row.
    gs = fig.add_gridspec(2, 2, height_ratios=[0.42, 0.58], width_ratios=[1.55, 1], hspace=0.06)

    sub_a = fig.add_subfigure(gs[0, :])
    sub_b = fig.add_subfigure(gs[1, 0])
    sub_c = fig.add_subfigure(gs[1, 1])

    ax_a = sub_a.add_subplot(111)
    ax_a.imshow(mpimg.imread(PANEL_A_PATH), aspect="equal")
    ax_a.set_axis_off()
    sub_a.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    panel_b_sitemap(sub_b)
    panel_c_excerpts(sub_c)

    for sub, label in [(sub_a, "a"), (sub_b, "b"), (sub_c, "c")]:
        sub.text(0.005, 0.985, label, fontsize=9, fontweight="bold", va="top", ha="left")

    fig.text(0.5, 0.006, PROVENANCE, fontsize=4.5, ha="center", va="bottom", color="#444444")

    save_fig(fig, "fig01_composite", gate=True)


if __name__ == "__main__":
    main()

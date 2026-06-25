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

PANEL_A_PATH = Path("/home/theo/brisa_paper/shared/figures/fig01_panelA.png")

# (site, patch_id, stl_path) — accented "maré" dir handled via PROJECT_ROOT join.
PANEL_C_EXCERPTS: list[tuple[str, str, Path]] = [
    ("rocinha", "ROC-P18", PROJECT_ROOT / "outputs" / "rocinha" / "print" / "ROC-P18_1to1000.stl"),
    ("vidigal", "VDG-P07", PROJECT_ROOT / "outputs" / "vidigal" / "print" / "VDG-P07_1to1000.stl"),
    ("maré", "MAR-P20", PROJECT_ROOT / "outputs" / "maré" / "print" / "MAR-P20_1to1000.stl"),
    (
        "riodaspedras",
        "RDP-P20",
        PROJECT_ROOT / "outputs" / "riodaspedras" / "print" / "RDP-P20_1to1000.stl",
    ),
]

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


def _render_excerpt(ax, site: str, patch_id: str, stl_path: Path) -> None:
    """Render one STL as a light-shaded massing model via plot_trisurf."""
    mesh = trimesh.load(stl_path)
    # Meshes are < 7k faces each, so no decimation needed; plot_trisurf is fast.
    verts = mesh.vertices
    faces = mesh.faces
    z = verts[:, 2]
    tri = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        z,
        triangles=faces,
        cmap="cividis",
        array=z[faces].mean(axis=1),
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    norm = (z[faces].mean(axis=1) - z.min()) / (np.ptp(z) + 1e-9)
    tri.set_facecolor(plt.get_cmap("cividis")(norm))

    ax.view_init(elev=35, azim=-60)
    ax.set_box_aspect(
        (np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2]) * 1.3 + 1e-9)
    )
    ax.set_axis_off()
    ax.set_title(f"{SITE_LABELS[site]} — {patch_id}", fontsize=5, pad=-2)


def panel_c_excerpts(subfig) -> None:
    """Render the 3-D massing excerpts in a 2x2 grid of mplot3d axes."""
    gs = subfig.add_gridspec(2, 2, hspace=0.05, wspace=0.05)
    for i, (site, patch_id, stl_path) in enumerate(PANEL_C_EXCERPTS):
        ax = subfig.add_subplot(gs[i // 2, i % 2], projection="3d")
        _render_excerpt(ax, site, patch_id, stl_path)


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

    save_fig(fig, "fig01_composite")


if __name__ == "__main__":
    main()

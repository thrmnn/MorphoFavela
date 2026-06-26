#!/usr/bin/env python3
"""Figure 0.1 — Conceptual framework and study sites.

Three-panel introductory figure for the journal manuscript:

  A. Conceptual schematic linking morphometric and terrain inputs through
     CFD and solar simulation to the four-state diagnostic taxonomy
     (stagnation x solar exposure → 2x2).
  B. Map of Rio de Janeiro showing the five study favelas coded by
     typology (hillside vs flatland).
  C. 3D-model excerpts illustrating the morphological range across sites
     (axonometric extrusions of a representative 200 m window per site).

Run:
    python docs/manuscript/figures/fig_0_1_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import box

# Reuse the canonical paper-figure style.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import (
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    add_north_arrow,
    add_scalebar,
    apply_style,
    clean_map_axes,
    hillshade,
    load_boundary,
    load_buildings,
    load_campaign_patches,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

# Manuscript-local typology binning. The global fig_style.SITE_TYPES marks
# Complexo do Alemão as "mixed"; per the Fig 0.1 spec it groups with hillside.
TYPOLOGY = {
    "vidigal": "hillside",
    "rocinha": "hillside",
    "complexo_do_alemao": "hillside",
    "riodaspedras": "flatland",
    "maré": "flatland",
}
TYPOLOGY_COLORS = {
    "hillside": "#CC6677",  # warm rose
    "flatland": "#44AA99",  # teal
}

# Representative campaign patches per site (chosen as the patch closest to
# the per-site median in (SVF, lambda_p)). Recomputed at script start so
# this list stays a hint, not a hard-coded contract.
DEFAULT_REPRESENTATIVE = {
    "vidigal": "VDG-P18",
    "rocinha": "ROC-P02",
    "complexo_do_alemao": "CDA-P08",
    "riodaspedras": "RDP-P14",
    "maré": "MAR-P07",
}

WINDOW_M = 150.0  # half-side of the 3D excerpt in metres → 150 m × 150 m

# Diagnostic taxonomy 2×2: rows = ventilation, cols = solar exposure.
TAXONOMY = {
    ("stagnant", "shaded"): {
        "label": "Dead-air\nshaded",
        "color": "#5d7eb3",  # cool blue-gray
    },
    ("stagnant", "exposed"): {
        "label": "Dead-air\nheat trap",
        "color": "#c0392b",  # red
    },
    ("ventilated", "shaded"): {
        "label": "Ventilated\nshaded",
        "color": "#3a8fb7",  # blue (best case)
    },
    ("ventilated", "exposed"): {
        "label": "Ventilated\nheat-exposed",
        "color": "#e8a33d",  # warm orange
    },
}

# Per-site one-line descriptors for Panel C captions.
SITE_DESCRIPTORS = {
    "vidigal": "steep hillslope, fine-grained",
    "rocinha": "valley + ridge, dense core",
    "complexo_do_alemao": "irregular hills, low-rise",
    "riodaspedras": "flat lowland, mid-rise",
    "maré": "flat bayside, regular grid",
}


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def pick_representative_patch(site: str) -> tuple[float, float, str]:
    """Centroid (x, y) and patch id of the patch closest to the median
    (SVF, λp) for the site. Falls back to the constant lookup if patches
    are missing."""
    try:
        df = load_campaign_patches(site)
    except FileNotFoundError:
        return None
    df = df.dropna(subset=["svf", "lambda_p", "center_x", "center_y"])
    if df.empty:
        return None
    rank = (df["svf"] - df["svf"].median()).abs() + (
        df["lambda_p"] - df["lambda_p"].median()
    ).abs()
    pick = df.loc[rank.idxmin()]
    return float(pick["center_x"]), float(pick["center_y"]), str(pick["patch_id"])


# ────────────────────────────────────────────────────────────────────────
# Panel A — conceptual schematic
# ────────────────────────────────────────────────────────────────────────


def _box(ax, xy, w, h, text, *, fc="#f4f4f4", ec="#333", fs=6.5, fw="normal"):
    rect = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.6,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=fw,
    )
    return rect


def _arrow(ax, xy_from, xy_to, *, label=None, lw=0.7, color="#333"):
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (xy_from[0] + xy_to[0]) / 2,
            (xy_from[1] + xy_to[1]) / 2 + 0.012,
            label,
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#444",
            style="italic",
        )


def draw_panel_a(ax):
    """Conceptual schematic in normalised [0,1] coordinates."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    clean_map_axes(ax)

    col_in_x, col_sim_x, col_out_x = 0.01, 0.30, 0.62
    col_w = 0.22
    box_h = 0.18

    # Inputs (3 stacked)
    inputs = [
        ("Morphometric grid\nλp · λf · σH · SVF", 0.74),
        ("Terrain\nslope · aspect · DTM", 0.50),
        ("Wind rose\n8 dir · Ū · calm", 0.26),
    ]
    in_centers = []
    for text, y in inputs:
        _box(ax, (col_in_x, y - box_h / 2), col_w, box_h, text, fc="#eef4fb", ec="#3a6fa8")
        in_centers.append((col_in_x + col_w, y))

    # Simulators (2 stacked)
    sims = [
        ("CFD k-ω SST\n250 m radius · 8 directions", 0.66),
        ("Solar envelope\nSOLWEIG / UMEP", 0.34),
    ]
    sim_left = []
    sim_right = []
    for text, y in sims:
        _box(ax, (col_sim_x, y - box_h / 2), col_w, box_h, text, fc="#fdf3e2", ec="#b07a23")
        sim_left.append((col_sim_x, y))
        sim_right.append((col_sim_x + col_w, y))

    # Inputs → simulators (unlabelled; data flow is described in the caption).
    edges = [
        (in_centers[0], sim_left[0]),
        (in_centers[0], sim_left[1]),
        (in_centers[1], sim_left[0]),
        (in_centers[1], sim_left[1]),
        (in_centers[2], sim_left[0]),
    ]
    for src, dst in edges:
        _arrow(ax, src, dst)

    # Output: 2x2 taxonomy grid drawn as a small mini-axes-like block.
    ox, oy = col_out_x, 0.18
    ow, oh = 0.36, 0.66
    cell_w, cell_h = ow / 2, (oh - 0.05) / 2  # leave room for axis labels
    grid_x0, grid_y0 = ox, oy + 0.05  # cells start above axis label area

    # Frame
    ax.add_patch(
        mpatches.Rectangle(
            (grid_x0, grid_y0),
            ow,
            oh - 0.05,
            facecolor="none",
            edgecolor="#222",
            linewidth=0.8,
        )
    )

    # Cells
    cells = [
        (("stagnant", "shaded"), 0, 1),
        (("stagnant", "exposed"), 1, 1),
        (("ventilated", "shaded"), 0, 0),
        (("ventilated", "exposed"), 1, 0),
    ]
    for (vent, sun), col, row in cells:
        meta = TAXONOMY[(vent, sun)]
        ax.add_patch(
            mpatches.Rectangle(
                (grid_x0 + col * cell_w, grid_y0 + row * cell_h),
                cell_w,
                cell_h,
                facecolor=meta["color"],
                edgecolor="#222",
                linewidth=0.4,
                alpha=0.85,
            )
        )
        ax.text(
            grid_x0 + col * cell_w + cell_w / 2,
            grid_y0 + row * cell_h + cell_h / 2,
            meta["label"],
            ha="center",
            va="center",
            fontsize=5.8,
            color="white",
            fontweight="bold",
        )

    # Axis labels for the taxonomy grid
    ax.text(
        grid_x0 + ow / 2,
        grid_y0 - 0.025,
        "solar exposure  →",
        ha="center",
        va="top",
        fontsize=6,
        style="italic",
    )
    ax.text(
        grid_x0 - 0.018,
        grid_y0 + (oh - 0.05) / 2,
        "ventilation  ↑",
        ha="right",
        va="center",
        fontsize=6,
        style="italic",
        rotation=90,
    )
    # Top label (the title-ish line for the diagnostic block)
    ax.text(
        grid_x0 + ow / 2,
        grid_y0 + (oh - 0.05) + 0.025,
        "Diagnostic taxonomy",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )

    # Simulators → taxonomy. CFD drives ventilation axis; solar drives sun axis.
    _arrow(ax, sim_right[0], (grid_x0, grid_y0 + (oh - 0.05) * 0.75))
    _arrow(ax, sim_right[1], (grid_x0, grid_y0 + (oh - 0.05) * 0.25))
    # Edge labels placed near the taxonomy-side of the simulator→output gap,
    # so they don't crowd the simulator-box text.
    sim_gap_x = col_out_x - 0.025
    ax.text(sim_gap_x, 0.71, "|U|, ACH", ha="right", va="center",
            fontsize=5.5, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.9))
    ax.text(sim_gap_x, 0.29, "hours of sun", ha="right", va="center",
            fontsize=5.5, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.9))

    # Column headers
    ax.text(col_in_x + col_w / 2, 0.95, "Inputs", ha="center", va="center",
            fontsize=7.5, fontweight="bold")
    ax.text(col_sim_x + col_w / 2, 0.95, "Simulators", ha="center", va="center",
            fontsize=7.5, fontweight="bold")
    ax.text(col_out_x + ow / 2, 0.95, "Output", ha="center", va="center",
            fontsize=7.5, fontweight="bold")


# ────────────────────────────────────────────────────────────────────────
# Panel B — Rio map by typology
# ────────────────────────────────────────────────────────────────────────


def draw_panel_b(ax):
    favelas_path = PROJECT_ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"
    rj_dtm_path = PROJECT_ROOT / "data" / "RJ" / "DTM_RJ.tif"
    if not favelas_path.exists():
        ax.text(0.5, 0.5, "Favelas_Limit_2019.shp missing",
                ha="center", va="center", transform=ax.transAxes)
        clean_map_axes(ax)
        return

    all_favelas = gpd.read_file(favelas_path)

    # Hillshade base
    if rj_dtm_path.exists():
        try:
            with rasterio.open(rj_dtm_path) as src:
                factor = 4
                dtm = src.read(1, out_shape=(src.height // factor, src.width // factor)).astype(float)
                if src.nodata is not None:
                    dtm[dtm == src.nodata] = np.nan
                dtm[np.abs(dtm) > 1e6] = np.nan
                bounds = src.bounds
                shade = hillshade(dtm, src.res[0] * factor)
                shade = np.where(np.isnan(dtm), 0.95, shade)
                ax.imshow(
                    shade, cmap="gray", vmin=0, vmax=1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    alpha=0.5, zorder=0,
                )
        except Exception as e:
            print(f"  hillshade failed: {e}")

    # All registered favelas in light grey
    all_favelas.plot(
        ax=ax, facecolor="none", edgecolor="#cccccc", linewidth=0.15, zorder=1,
    )

    # Campaign sites coloured by typology
    boundaries = {}
    for site in SITE_ORDER:
        try:
            boundaries[site] = load_boundary(site)
        except FileNotFoundError:
            for cand in [site.upper(), site.replace("_", " ").upper()]:
                m = all_favelas[all_favelas["nome"].str.upper().str.contains(cand, na=False)]
                if not m.empty:
                    boundaries[site] = m
                    break

    for site, gdf in boundaries.items():
        col = TYPOLOGY_COLORS[TYPOLOGY[site]]
        gdf.plot(
            ax=ax, facecolor=col, edgecolor=col,
            alpha=0.6, linewidth=0.8, zorder=3,
        )
        c = gdf.geometry.union_all().centroid
        ax.annotate(
            SITE_LABELS[site],
            xy=(c.x, c.y),
            fontsize=5.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      alpha=0.85, edgecolor="none"),
            zorder=5,
        )

    # Zoom to the bounding box of the campaign sites
    arr = np.array([gdf.total_bounds for gdf in boundaries.values()])
    if len(arr):
        pad = 5000
        ax.set_xlim(arr[:, 0].min() - pad, arr[:, 2].max() + pad)
        ax.set_ylim(arr[:, 1].min() - pad, arr[:, 3].max() + pad)

    clean_map_axes(ax)
    ax.set_aspect("equal")
    add_scalebar(ax, length_m=5000)
    add_north_arrow(ax)

    # Legend
    handles = [
        mpatches.Patch(facecolor=TYPOLOGY_COLORS["hillside"], edgecolor="none",
                       alpha=0.7, label="Hillside"),
        mpatches.Patch(facecolor=TYPOLOGY_COLORS["flatland"], edgecolor="none",
                       alpha=0.7, label="Flatland"),
    ]
    ax.legend(
        handles=handles, loc="lower right", frameon=True,
        framealpha=0.9, edgecolor="none", fontsize=6, title="Typology",
        title_fontsize=6.5,
    )


# ────────────────────────────────────────────────────────────────────────
# Panel C — axonometric 3D excerpts
# ────────────────────────────────────────────────────────────────────────


def _extrude_buildings(ax3d, gdf: gpd.GeoDataFrame, z0_grid: float):
    """Add Poly3DCollection extrusions for every polygon in `gdf`.

    Uses `base` for the bottom z and `topo` for the top z; falls back to
    `altura` + z0_grid when those are missing.
    """
    polys_top = []
    polys_bot = []
    polys_side = []
    for geom, base, topo, altura in zip(
        gdf.geometry, gdf.get("base"), gdf.get("topo"), gdf.get("altura")
    ):
        if geom is None or geom.is_empty:
            continue
        # Resolve z bounds robustly.
        if pd_is_finite(base) and pd_is_finite(topo) and topo > base:
            zb, zt = float(base), float(topo)
        elif pd_is_finite(altura) and altura > 0:
            zb = z0_grid
            zt = z0_grid + float(altura)
        else:
            continue

        rings = (
            [geom.exterior] if geom.geom_type == "Polygon"
            else [g.exterior for g in geom.geoms]
        )
        for ring in rings:
            xs, ys = ring.coords.xy
            xs = list(xs)
            ys = list(ys)
            top = [(x, y, zt) for x, y in zip(xs, ys)]
            bot = [(x, y, zb) for x, y in zip(xs, ys)]
            polys_top.append(top)
            polys_bot.append(bot)
            for i in range(len(xs) - 1):
                polys_side.append([
                    (xs[i], ys[i], zb),
                    (xs[i + 1], ys[i + 1], zb),
                    (xs[i + 1], ys[i + 1], zt),
                    (xs[i], ys[i], zt),
                ])

    if polys_side:
        ax3d.add_collection3d(Poly3DCollection(
            polys_side, facecolors="#bcbcbc", edgecolors="#3a3a3a",
            linewidths=0.15, alpha=0.95,
        ))
    if polys_top:
        ax3d.add_collection3d(Poly3DCollection(
            polys_top, facecolors="#dcdcdc", edgecolors="#2a2a2a",
            linewidths=0.2, alpha=0.98,
        ))


def pd_is_finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _terrain_surface(ax3d, dtm_path: Path, cx: float, cy: float, half: float):
    """Draw a low-detail terrain mesh under the buildings."""
    if not dtm_path.exists():
        return None
    with rasterio.open(dtm_path) as src:
        # Window in pixel space
        inv = ~src.transform
        col0, row0 = inv * (cx - half, cy + half)
        col1, row1 = inv * (cx + half, cy - half)
        col0, col1 = sorted([int(col0), int(col1)])
        row0, row1 = sorted([int(row0), int(row1)])
        col0 = max(col0, 0)
        row0 = max(row0, 0)
        col1 = min(col1, src.width)
        row1 = min(row1, src.height)
        if col1 <= col0 or row1 <= row0:
            return None
        win = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        # Decimate to ~60 samples per side max for speed
        out_h = min(60, win.height)
        out_w = min(60, win.width)
        z = src.read(1, window=win, out_shape=(out_h, out_w)).astype(float)
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        z[np.abs(z) > 1e6] = np.nan

    # Build coordinate grid in world space
    xs = np.linspace(cx - half, cx + half, z.shape[1])
    ys = np.linspace(cy + half, cy - half, z.shape[0])
    X, Y = np.meshgrid(xs, ys)

    # Median z anchors building bases when `base` is missing.
    z_med = float(np.nanmedian(z)) if np.isfinite(np.nanmedian(z)) else 0.0

    ax3d.plot_surface(
        X, Y, np.where(np.isnan(z), z_med, z),
        rstride=2, cstride=2,
        cmap="Greys", vmin=np.nanmin(z) - 5, vmax=np.nanmax(z) + 20,
        linewidth=0, antialiased=False, alpha=0.55, zorder=0,
    )
    return z_med


def draw_panel_c_one(ax3d, site: str):
    """Render a single 3D excerpt for `site` on the supplied 3-D axes."""
    rep = pick_representative_patch(site)
    if rep is None:
        ax3d.text2D(0.5, 0.5, f"{site}: no patches", transform=ax3d.transAxes,
                    ha="center", va="center")
        return
    cx, cy, _pid = rep

    # Buildings within the window
    bld = load_buildings(site, extended=True)
    win = box(cx - WINDOW_M, cy - WINDOW_M, cx + WINDOW_M, cy + WINDOW_M)
    sub = bld[bld.intersects(win)].copy()
    sub["geometry"] = sub.geometry.intersection(win)
    sub = sub[~sub.geometry.is_empty]

    # Terrain
    dtm_path = PROJECT_ROOT / "data" / site / "dtm_extended_300m.tif"
    z_med = _terrain_surface(ax3d, dtm_path, cx, cy, WINDOW_M)
    if z_med is None:
        z_med = 0.0

    # Buildings
    _extrude_buildings(ax3d, sub, z0_grid=z_med)

    # Cosmetic axes
    ax3d.set_xlim(cx - WINDOW_M, cx + WINDOW_M)
    ax3d.set_ylim(cy - WINDOW_M, cy + WINDOW_M)
    # Z range: tie to terrain ± buildings, with vertical exaggeration so
    # the 3D illusion reads at thumbnail scale.
    ax3d.set_zlim(z_med - 8, z_med + 90)
    try:
        ax3d.set_box_aspect((1, 1, 0.55))
    except AttributeError:
        pass
    ax3d.view_init(elev=38, azim=-58)
    ax3d.set_axis_off()
    ax3d.set_proj_type("ortho")

    # Caption: site name as the axes title (clipped to the panel,
    # avoids running into neighbouring thumbnails).
    ax3d.set_title(
        SITE_LABELS[site],
        fontsize=6.5,
        fontweight="bold",
        color=TYPOLOGY_COLORS[TYPOLOGY[site]],
        pad=1,
    )
    # Descriptor as a small line below the panel.
    ax3d.text2D(
        0.5, -0.04,
        SITE_DESCRIPTORS[site],
        transform=ax3d.transAxes,
        fontsize=5.5,
        ha="center",
        va="top",
        style="italic",
        color="#444",
    )


# ────────────────────────────────────────────────────────────────────────
# Compose
# ────────────────────────────────────────────────────────────────────────


def main():
    apply_style()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.95))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.66, 0.34], hspace=0.10,
    )

    # Top row: A (left, schematic) + B (right, Rio map)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.58, 0.42], wspace=0.06)

    ax_a = fig.add_subplot(top[0])
    draw_panel_a(ax_a)
    ax_a.text(-0.01, 1.02, "A", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="bottom")

    ax_b = fig.add_subplot(top[1])
    draw_panel_b(ax_b)
    ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="bottom")

    # Bottom row: C is a horizontal strip of 5 axonometric thumbnails.
    inner_c = outer[1].subgridspec(1, 5, wspace=0.04)
    for i, site in enumerate(SITE_ORDER):
        ax3d = fig.add_subplot(inner_c[i], projection="3d")
        draw_panel_c_one(ax3d, site)
        if i == 0:
            ax3d.text2D(-0.02, 1.04, "C", transform=ax3d.transAxes,
                        fontsize=10, fontweight="bold", va="bottom")

    out_png = EXPORTS_DIR / "fig_0_1_overview.png"
    out_svg = EXPORTS_DIR / "fig_0_1_overview.svg"
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"  wrote {out_svg}")


if __name__ == "__main__":
    main()

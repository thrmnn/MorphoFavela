#!/usr/bin/env python3
"""Figure 0.2 — Morphometric and terrain landscape across favelas.

Two-block descriptive figure for the journal manuscript:

  A. Six-panel choropleth grid of Vidigal (anchor site) at 10 m resolution:
     SVF, lambda_p, porosity, lambda_f, sigma_H, slope. lambda_p and
     porosity are placed adjacently to expose the redundancy
     ( rho(lambda_p, porosity) approx -0.97 in Vidigal ).
  B. Cross-favela ridge plots: one row per indicator, five offset
     KDE ridges per row (one per site, ordered hillside to flatland).
     Threshold lines mark the SVF stratification cuts (0.15, 0.30),
     the lambda_p threshold (0.5), and the slope threshold (15 deg).

Aspect / northness lives in a separate ED figure.

Run:
    python docs/manuscript/figures/fig_0_2_landscape.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
from fig_style import (
    SITE_COLORS,
    SITE_LABELS,
    SITE_ORDER,
    WIDTH_DOUBLE,
    apply_style,
    clean_map_axes,
    load_boundary,
    load_buildings,
)

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

ANCHOR_SITE = "vidigal"

# ---------------------------------------------------------------------------
# Indicator metadata. Order is the visual order in Panel A (row-major 2x3)
# and the row order in Panel B.
# lambda_p and porosity sit next to each other so the redundancy reads
# at a glance.
# ---------------------------------------------------------------------------
INDICATORS = [
    # (column, label, units, cmap, vmin, vmax, x_range_for_ridge, threshold_lines)
    ("svf",           "SVF",                "",       "cividis",   0.0, 1.0,  (0.0, 1.0),  [0.15, 0.30]),
    ("lambda_p",      r"$\lambda_p$",       "",       "magma",     0.0, 1.0,  (0.0, 1.0),  [0.5]),
    ("porosity",      "Porosity",           "",       "cividis_r", 0.0, 1.0,  (0.0, 1.0),  []),
    ("lambda_f_mean", r"$\lambda_f$",       "",       "inferno",   0.0, 10.0, (0.0, 12.0), []),
    ("sigma_h",       r"$\sigma_H$",        "(m)",    "magma",     0.0, 6.0,  (0.0, 6.0),  []),
    ("slope_deg",     "Slope",              "(°)",    None,        0.0, 60.0, (0.0, 60.0), [15.0]),
]

# Custom slope colormap — truncated `terrain` so the bottom is green
# (flat ground), not the blue water band. Anchored 0-60 deg.
_terrain = plt.get_cmap("terrain")
SLOPE_CMAP = mpl.colors.ListedColormap(_terrain(np.linspace(0.25, 1.0, 256)))
for spec in INDICATORS:
    pass  # placeholder — assignment done below


def _resolve_cmap(name):
    return SLOPE_CMAP if name is None else plt.get_cmap(name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_grid(site: str) -> gpd.GeoDataFrame:
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    return gpd.read_file(path)


def _load_all_grids():
    return {site: _load_grid(site) for site in SITE_ORDER}


# ---------------------------------------------------------------------------
# Panel A: Vidigal choropleth grid
# ---------------------------------------------------------------------------
def draw_panel_a(fig, gs_a, grid: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame,
                 boundary: gpd.GeoDataFrame):
    """Six maps of the anchor site, 2 rows x 3 cols. Returns list of axes."""
    panel_grid = gs_a.subgridspec(2, 3, hspace=0.32, wspace=0.18)

    # Common bounds for all six maps so they line up exactly.
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 50.0  # m

    axes = []
    for idx, (col, label, units, cmap_name, vmin, vmax, _xr, _thr) in enumerate(INDICATORS):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(panel_grid[r, c])
        axes.append(ax)
        cmap = _resolve_cmap(cmap_name)

        # Building footprints as a faint backdrop for spatial context.
        buildings.plot(ax=ax, facecolor="#dddddd", edgecolor="none",
                       linewidth=0.0, zorder=1)

        # Cells with values.
        gd = grid.dropna(subset=[col])
        gd.plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vmax,
                linewidth=0.0, alpha=0.85, zorder=2)

        # Boundary outline.
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.5,
                               linestyle="--", zorder=3)

        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        clean_map_axes(ax)

        title = f"{label} {units}".strip()
        ax.set_title(title, fontsize=7, pad=3)

        # Horizontal colorbar below each map.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="4%", pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin, vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5, length=2, width=0.3, pad=1)
        # Just two ticks (vmin, vmax) for compactness; add midpoint where useful.
        if vmax <= 1.5:
            cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        else:
            cb.set_ticks([vmin, vmax])

    # Overall panel label (top-left of the first map).
    pos = axes[0].get_position()
    fig.text(pos.x0 - 0.012, pos.y1 + 0.012,
             "a", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(pos.x0 + 0.018, pos.y1 + 0.012,
             f"Morphometric maps — {SITE_LABELS[ANCHOR_SITE]} (10 m grid)",
             fontsize=7.5, va="bottom", ha="left")
    return axes


# ---------------------------------------------------------------------------
# Panel B: cross-site ridge plots
# ---------------------------------------------------------------------------
def _draw_kde_cell(ax, values: np.ndarray, x_range: tuple[float, float],
                   color: str, thresholds: list[float],
                   peak_h: float = 0.78, alpha: float = 0.55) -> None:
    """One cell: filled KDE + median tick + IQR/5-95 baseline strip."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    # KDE in the cell's x-range.
    if len(v) >= 10:
        v_in = v[(v >= x_range[0]) & (v <= x_range[1])]
        if len(v_in) >= 10:
            try:
                kde = gaussian_kde(v_in, bw_method=0.18)
                xs = np.linspace(*x_range, 220)
                ys = kde(xs)
                if ys.max() > 0:
                    ys_norm = (ys / ys.max()) * peak_h
                    # Lift everything above the strip baseline.
                    base = 0.18
                    ax.fill_between(xs, base, base + ys_norm, color=color,
                                    alpha=alpha, lw=0, zorder=3)
                    ax.plot(xs, base + ys_norm, color=color,
                            lw=0.7, zorder=4)
            except np.linalg.LinAlgError:
                pass

    # Summary-statistics strip at the baseline (dark grey, Tufte-ish).
    if len(v) >= 5:
        q05, q25, med, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        strip_y = 0.07
        # 5-95 percentile thin line.
        ax.plot([q05, q95], [strip_y, strip_y],
                color="#444444", lw=0.5, zorder=5, solid_capstyle="butt")
        # 25-75 percentile thicker bar.
        ax.plot([q25, q75], [strip_y, strip_y],
                color="#222222", lw=1.6, zorder=6, solid_capstyle="butt")
        # Median tick (small vertical hairline).
        ax.plot([med, med], [strip_y - 0.03, strip_y + 0.03],
                color="white", lw=0.9, zorder=7, solid_capstyle="butt")

    # Threshold reference lines (dashed grey, drawn through the cell).
    for t in thresholds:
        ax.axvline(t, color="#666666", linestyle=(0, (3, 2)), linewidth=0.5,
                   zorder=2, alpha=0.9)

    ax.set_xlim(*x_range)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.tick_params(axis="x", labelsize=5, length=1.8, width=0.3, pad=1)


def draw_panel_b(fig, gs_b, grids: dict[str, gpd.GeoDataFrame]):
    """5 sites (rows) x 6 indicators (cols) small-multiples grid.

    Row = favela, column = indicator. Scanning a column gives a direct
    cross-site comparison for that indicator (the analytical question).
    Scanning a row gives a single favela's morphometric profile.
    """
    n_ind = len(INDICATORS)
    n_site = len(SITE_ORDER)

    block = gs_b.subgridspec(
        n_site, n_ind,
        hspace=0.55, wspace=0.18,
    )

    first_in_row = {}   # leftmost cell per site, for site label anchor
    first_in_col = {}   # topmost cell per indicator, for column header
    last_in_col = {}    # bottommost cell per indicator, for porosity ρ tag

    for r, site in enumerate(SITE_ORDER):
        for c, (col, label, units, _cm, _vmin, _vmax, x_range, thr) in enumerate(INDICATORS):
            ax = fig.add_subplot(block[r, c])
            data = grids[site][col].dropna().to_numpy()
            _draw_kde_cell(ax, data, x_range,
                           color=SITE_COLORS[site], thresholds=thr)

            # Left column → site name as row label, plus a small colored
            # swatch to keep site identity unambiguous at this size.
            if c == 0:
                first_in_row[r] = ax
                ax.text(-0.30, 0.5, SITE_LABELS[site],
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=7.5, fontweight="bold", color="#222222")
                # Color swatch (small filled rectangle just left of the cell).
                ax.add_patch(plt.Rectangle(
                    (-0.08, 0.20), 0.025, 0.60,
                    transform=ax.transAxes, clip_on=False,
                    facecolor=SITE_COLORS[site], edgecolor="none",
                    zorder=10,
                ))

            # Top row → indicator title as column header.
            if r == 0:
                first_in_col[c] = ax
                title_txt = f"{label} {units}".strip()
                ax.set_title(title_txt, fontsize=7.5, pad=18,
                             color="#222222", fontweight="bold")

                # Threshold values printed just under the column title,
                # so the analytical anchors (0.15 / 0.30 / 0.5 / 15)
                # stay visible. Stagger vertical position when multiple
                # thresholds sit close together (e.g. SVF at 0.15 + 0.30).
                for i, t in enumerate(thr):
                    y_base = 1.04 if i % 2 == 0 else 1.13
                    ax.text(t, y_base, f"{t:g}",
                            transform=ax.get_xaxis_transform(),
                            fontsize=4.8, color="#555555",
                            ha="center", va="bottom", clip_on=False,
                            bbox=dict(boxstyle="round,pad=0.08",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.95))

            if r == n_site - 1:
                last_in_col[c] = ax

    # ρ annotation under the porosity column — anchored to the bottommost
    # porosity cell so it sits in clear margin and stays close to its row.
    porosity_col = next(i for i, ind in enumerate(INDICATORS) if ind[0] == "porosity")
    r_anchor = pearsonr(grids[ANCHOR_SITE]["lambda_p"],
                        grids[ANCHOR_SITE]["porosity"])[0]
    pos_bot = last_in_col[porosity_col].get_position()
    fig.text(pos_bot.x0 + pos_bot.width * 0.5, pos_bot.y0 - 0.028,
             rf"$\rho(\lambda_p,\,\mathrm{{porosity}})\approx{r_anchor:.2f}$",
             ha="center", va="top", fontsize=5.8, style="italic",
             color="#333",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                       edgecolor="#bbbbbb", linewidth=0.3, alpha=0.95))

    # Panel label, anchored above the first cell of the first row.
    pos = first_in_row[0].get_position()
    fig.text(pos.x0 - 0.080, pos.y1 + 0.040,
             "b", fontsize=10, fontweight="bold", va="bottom", ha="left")
    fig.text(pos.x0 - 0.055, pos.y1 + 0.040,
             "Cross-favela distributions  (white tick = median, "
             "black bar = 25–75%, line = 5–95%)",
             fontsize=7.5, va="bottom", ha="left")

    return list(first_in_row.values())


# Site legend is no longer needed — column headers in Panel B's grid now
# carry site identity in colored text.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_style()
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

    print("Loading grids ...")
    grids = _load_all_grids()
    print(f"  Vidigal: {len(grids[ANCHOR_SITE])} cells")

    print(f"Loading {ANCHOR_SITE} buildings + boundary ...")
    buildings = load_buildings(ANCHOR_SITE, extended=False)
    boundary = load_boundary(ANCHOR_SITE)

    fig = plt.figure(figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.10))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.58, 0.42],
        hspace=0.20,
        left=0.13, right=0.98, top=0.95, bottom=0.05,
    )
    gs_a, gs_b = outer[0], outer[1]

    print("Drawing Panel A ...")
    a_axes = draw_panel_a(fig, gs_a, grids[ANCHOR_SITE], buildings, boundary)

    print("Drawing Panel B ...")
    b_axes = draw_panel_b(fig, gs_b, grids)


    out_png = EXPORTS_DIR / "fig_0_2_landscape.png"
    out_svg = EXPORTS_DIR / "fig_0_2_landscape.svg"
    print(f"Saving {out_png.name} + {out_svg.name} ...")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

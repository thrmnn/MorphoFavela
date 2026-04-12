#!/usr/bin/env python3
"""Figure S3: Resolution sensitivity — 10m vs 20m morphometric grids.

Generates three variants for evaluation:

    figS3a_distribution_overlay.{svg,png}
        5 sites × 5 indicators, KDE curves 10m vs 20m overlaid.
        Shows whether aggregation shifts distributions.

    figS3b_scatter_comparison.{svg,png}
        For Vidigal: upscale 10m cells to 20m bins, scatter against native 20m.
        Per-indicator agreement plot with 1:1 line and Pearson r.

    figS3c_difference_map.{svg,png}
        Vidigal: native 10m maps, resampled 20m maps, |difference| maps.
        Side-by-side for SVF, λp, slope.

Pick the variant(s) that best support the manuscript narrative.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde


INDICATORS = [
    ("svf",           r"$SVF$",           (0, 1)),
    ("lambda_p",      r"$\lambda_p$",     (0, 1)),
    ("lambda_f_mean", r"$\lambda_f$",     (0, 2)),
    ("sigma_h",       r"$\sigma_H$ (m)",  (0, 8)),
    ("slope_deg",     "Slope (°)",         (0, 50)),
]


def load_grid_res(site: str, suffix: str = "") -> gpd.GeoDataFrame | None:
    """Load a grid by resolution suffix. '' = 10m default, '_20m' = 20m."""
    path = PROJECT_ROOT / "outputs" / site / f"morphometrics{suffix}" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        return None
    return gpd.read_file(path)


# ══════════════════════════════════════════════════════════════════════
#  Variant A: distribution overlay
# ══════════════════════════════════════════════════════════════════════


def make_variant_a():
    apply_style()

    fig, axes = plt.subplots(
        len(SITE_ORDER), len(INDICATORS),
        figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.72),
        sharex="col",
    )

    for row, site in enumerate(SITE_ORDER):
        g10 = load_grid_res(site)
        g20 = load_grid_res(site, "_20m")
        if g10 is None or g20 is None:
            continue

        for col, (column, label, xlim) in enumerate(INDICATORS):
            ax = axes[row, col]
            v10 = g10[column].dropna().values
            v20 = g20[column].dropna().values

            if len(v10) < 5 or len(v20) < 5:
                ax.set_visible(False)
                continue

            try:
                x = np.linspace(xlim[0], xlim[1], 200)
                k10 = gaussian_kde(v10, bw_method=0.1)
                k20 = gaussian_kde(v20, bw_method=0.1)
                ax.fill_between(x, k10(x), color=SITE_COLORS[site], alpha=0.12, linewidth=0)
                ax.plot(x, k10(x), color=SITE_COLORS[site], linewidth=0.8, label="10 m")
                ax.plot(x, k20(x), color="#333333", linewidth=0.8, linestyle="--", label="20 m")
            except Exception:
                ax.set_visible(False)
                continue

            ax.set_xlim(xlim)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)

            if row == 0:
                ax.set_title(label, fontsize=7, pad=4)
            if col == 0:
                ax.set_ylabel(SITE_LABELS[site], fontsize=6, rotation=0,
                              labelpad=30, va="center", ha="right")
            if row == len(SITE_ORDER) - 1:
                ax.tick_params(labelsize=5)
            else:
                ax.set_xticklabels([])

            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=5, frameon=False)

    fig.tight_layout(w_pad=0.8, h_pad=0.4)
    save_fig(fig, "figS3a_distribution_overlay")


# ══════════════════════════════════════════════════════════════════════
#  Variant B: 20m vs upscaled-10m scatter (Vidigal)
# ══════════════════════════════════════════════════════════════════════


def _upscale_10m_to_20m(g10: gpd.GeoDataFrame, g20: gpd.GeoDataFrame, col: str) -> pd.Series:
    """Spatial average of 10m cell values within each 20m cell footprint."""
    # Each 20m cell contains up to 4 10m cells. Match by centroid containment.
    from scipy.spatial import cKDTree
    tree = cKDTree(g10[["centroid_x", "centroid_y"]].values)
    # For each 20m centroid, query points within a 10m radius (captures ~4 cells)
    coords_20 = g20[["centroid_x", "centroid_y"]].values
    idxs = tree.query_ball_point(coords_20, r=10.0)
    v10 = g10[col].values
    averaged = np.array([np.nanmean(v10[i]) if i else np.nan for i in idxs])
    return pd.Series(averaged, index=g20.index)


def make_variant_b():
    apply_style()

    g10 = load_grid_res("vidigal")
    g20 = load_grid_res("vidigal", "_20m")
    if g10 is None or g20 is None:
        print("  SKIP variant B — Vidigal grids missing.")
        return

    fig, axes = plt.subplots(1, len(INDICATORS),
                             figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.22))

    for ax, (col, label, lim) in zip(axes, INDICATORS):
        upscaled = _upscale_10m_to_20m(g10, g20, col)
        native = g20[col].values
        valid = ~(np.isnan(upscaled.values) | np.isnan(native))
        u, n = upscaled.values[valid], native[valid]
        if len(u) < 5:
            ax.set_visible(False)
            continue

        ax.scatter(u, n, s=3, c=SITE_COLORS["vidigal"], alpha=0.35, linewidths=0,
                   rasterized=True)
        ax.plot(lim, lim, color="#888888", linewidth=0.5, linestyle="--")

        r, _ = stats.pearsonr(u, n)
        stat_annotation(ax, f"$r$ = {r:.2f}\n$n$ = {len(u):,d}", loc="upper left")

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"10 m → 20 m avg", fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("Native 20 m", fontsize=6)
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_aspect("equal")

    fig.tight_layout(w_pad=0.6)
    save_fig(fig, "figS3b_scatter_comparison")


# ══════════════════════════════════════════════════════════════════════
#  Variant C: difference maps (Vidigal)
# ══════════════════════════════════════════════════════════════════════


def make_variant_c():
    apply_style()

    g10 = load_grid_res("vidigal")
    g20 = load_grid_res("vidigal", "_20m")
    if g10 is None or g20 is None:
        print("  SKIP variant C — Vidigal grids missing.")
        return

    map_indicators = [
        ("svf",       r"$SVF$",        CMAP_SVF,      (0, 0.8)),
        ("lambda_p",  r"$\lambda_p$",  CMAP_LAMBDA_P, (0, 1.0)),
        ("slope_deg", "Slope (°)",      CMAP_SLOPE,    (0, 45)),
    ]

    n_rows = len(map_indicators)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 1.05))

    for row, (col, label, cmap, vlim) in enumerate(map_indicators):
        # 10m native
        ax = axes[row, 0]
        g10.plot(ax=ax, column=col, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                 edgecolor="none", legend=False)
        clean_map_axes(ax)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title("10 m native", fontsize=7, pad=4)
        ax.set_ylabel(label, fontsize=7, rotation=90, labelpad=8)
        add_scalebar(ax)

        # 20m native
        ax = axes[row, 1]
        g20.plot(ax=ax, column=col, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                 edgecolor="none", legend=False)
        clean_map_axes(ax)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title("20 m native", fontsize=7, pad=4)
        add_scalebar(ax)

        # Difference map
        ax = axes[row, 2]
        upscaled = _upscale_10m_to_20m(g10, g20, col)
        g20_diff = g20.copy()
        g20_diff["_diff"] = np.abs(g20[col].values - upscaled.values)
        dmax = float(np.nanpercentile(g20_diff["_diff"].values, 95))
        g20_diff.plot(ax=ax, column="_diff", cmap="Reds", vmin=0, vmax=dmax,
                      edgecolor="none", legend=False)
        clean_map_axes(ax)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title("|20 m − 10 m upscaled|", fontsize=7, pad=4)
        add_scalebar(ax)

        # Colorbar for native maps
        import matplotlib.colors as mcolors
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=mcolors.Normalize(vmin=vlim[0], vmax=vlim[1]))
        cbar = fig.colorbar(sm, ax=axes[row, :2].tolist(), shrink=0.7, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=5, width=0.3, length=2)
        cbar.outline.set_linewidth(0.3)

        sm_d = plt.cm.ScalarMappable(cmap="Reds",
                                     norm=mcolors.Normalize(vmin=0, vmax=dmax))
        cbar_d = fig.colorbar(sm_d, ax=ax, shrink=0.7, pad=0.02, aspect=20)
        cbar_d.ax.tick_params(labelsize=5, width=0.3, length=2)
        cbar_d.outline.set_linewidth(0.3)

    save_fig(fig, "figS3c_difference_map")


def main():
    print("Generating Fig S3 variants...")
    make_variant_a()
    make_variant_b()
    make_variant_c()


if __name__ == "__main__":
    main()

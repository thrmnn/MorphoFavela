"""Publication-quality figures for morphometric audit reports.

All figures follow a consistent Swiss-academic design language:
- Perceptually uniform colormaps (viridis, magma, cividis)
- 300 DPI raster output
- Consistent spatial extent and north orientation
- Scale bar, north arrow, coordinate labels
- Clean typography (sans-serif)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm

from src.cartography import (
    add_north_arrow,
    add_scale_bar,
    add_terrain_contours,
    apply_publication_style,
    format_utm_axes,
)

logger = logging.getLogger(__name__)

# -- Swiss design palette --
SWISS = {
    "primary": "#1A1A1A",
    "secondary": "#5C5C5C",
    "accent": "#2A5FA5",
    "background": "#FFFFFF",
    "light_gray": "#F2F2F2",
    "grid": "#D9D9D9",
    "caption": "#666666",
    "building_face": "#D4D4D4",
    "building_edge": "#888888",
}

DPI = 300


def _hillshade(dem, azimuth=315, altitude=45):
    """Compute hillshade from a DEM array."""
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    shade = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(
        az_rad - aspect
    )
    return np.clip(shade, 0, 1)


def _add_hillshade_background(ax, dtm_path: Path):
    """Render hillshade DTM as a background image on *ax* and return extent."""
    with rasterio.open(dtm_path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        hs = _hillshade(dem)
        ax.imshow(hs, extent=extent, cmap="Greys_r", alpha=0.6, origin="upper")
    return extent


def _set_bounds_from_gdf(ax, gdf: gpd.GeoDataFrame, margin: float = 20):
    """Set axes limits to the total bounds of *gdf* with a margin."""
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)


# =========================================================================
# Figure 1 -- Site Overview
# =========================================================================


def figure_site_overview(
    buildings: gpd.GeoDataFrame,
    dtm_path: Path,
    output_path: Path,
    boundary: Optional[gpd.GeoDataFrame] = None,
    area_name: str = "Study Area",
):
    """Figure 1: Site overview -- buildings on hillshade DTM."""
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Hillshade background
    _add_hillshade_background(ax, dtm_path)

    # Terrain contours via cartography module
    add_terrain_contours(ax, dtm_path)

    # Building footprints
    buildings.plot(
        ax=ax,
        facecolor=SWISS["building_face"],
        edgecolor=SWISS["building_edge"],
        linewidth=0.3,
        alpha=0.85,
    )

    # Boundary
    if boundary is not None:
        boundary.boundary.plot(
            ax=ax, color=SWISS["accent"], linewidth=1.5, linestyle="--", alpha=0.8
        )

    ax.set_title(
        f"{area_name} — Site Overview",
        fontsize=14,
        fontweight="bold",
        color=SWISS["primary"],
        pad=12,
    )

    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)

    # Building count annotation
    ax.text(
        0.02,
        0.02,
        f"{len(buildings):,} buildings",
        transform=ax.transAxes,
        fontsize=8,
        color=SWISS["secondary"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.8,
            edgecolor=SWISS["grid"],
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 1 saved: %s", output_path)


# =========================================================================
# Figure 2 -- SVF Map
# =========================================================================


def figure_svf_map(
    grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    output_path: Path,
    area_name: str = "Study Area",
    svf_col: str = "svf",
):
    """Figure 2: SVF map on 10m grid with diverging colormap at SVF=0.15."""
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Filter cells with SVF data
    valid = grid[grid[svf_col].notna()].copy()
    if valid.empty:
        logger.warning("No SVF data for map; skipping.")
        plt.close(fig)
        return

    # Diverging colormap centered at 0.15
    vmin, vmax = 0.0, min(1.0, valid[svf_col].quantile(0.99))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.15, vmax=vmax)

    # Buildings background
    buildings.plot(
        ax=ax,
        facecolor=SWISS["building_face"],
        edgecolor=SWISS["building_edge"],
        linewidth=0.2,
        alpha=0.4,
    )

    # SVF grid
    valid.plot(
        column=svf_col,
        ax=ax,
        cmap="RdYlBu",
        norm=norm,
        alpha=0.75,
        edgecolor="none",
        legend=True,
        legend_kwds={
            "label": "Sky View Factor",
            "shrink": 0.6,
            "pad": 0.02,
        },
    )

    # Annotate extreme areas
    very_low = valid[valid[svf_col] < 0.10]
    if len(very_low) > 0:
        centroid = very_low.geometry.union_all().centroid
        ax.annotate(
            f"SVF < 0.10\n({len(very_low)} cells)",
            xy=(centroid.x, centroid.y),
            fontsize=7,
            color="#b91c1c",
            fontweight="bold",
            ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_title(
        f"{area_name} — Sky View Factor",
        fontsize=14,
        fontweight="bold",
        color=SWISS["primary"],
        pad=12,
    )

    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 2 saved: %s", output_path)


# =========================================================================
# Figure 3 -- Morphometric Panel (2x3)
# =========================================================================


def figure_morphometric_panel(
    grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    output_path: Path,
    area_name: str = "Study Area",
):
    """Figure 3: 2x3 panel of morphometric indicators."""
    apply_publication_style()

    panels = [
        ("lambda_p", r"$\lambda_p$ (Plan Area Density)", "YlOrRd"),
        ("lambda_f_mean", r"$\lambda_f$ (Frontal Area Density)", "OrRd"),
        ("porosity", "Porosity", "YlGnBu"),
        ("sigma_h", r"$\sigma_H$ (Height Variability, m)", "YlOrBr"),
        ("street_orientation_entropy", "Street Orientation Entropy", "PuBuGn"),
        ("slope_deg", "Terrain Slope (degrees)", "terrain"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (col, label, cmap) in enumerate(panels):
        ax = axes[idx]

        # Buildings background
        buildings.plot(
            ax=ax,
            facecolor=SWISS["building_face"],
            edgecolor="none",
            linewidth=0,
            alpha=0.25,
        )

        if col in grid.columns:
            valid = grid[grid[col].notna()]
            if not valid.empty:
                valid.plot(
                    column=col,
                    ax=ax,
                    cmap=cmap,
                    alpha=0.8,
                    edgecolor="none",
                    legend=True,
                    legend_kwds={"shrink": 0.5, "pad": 0.02},
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=SWISS["secondary"],
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Not computed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=SWISS["secondary"],
            )

        ax.set_title(label, fontsize=10, fontweight="bold", color=SWISS["primary"])

        # Synchronize spatial extent
        _set_bounds_from_gdf(ax, buildings)

        # UTM formatting (hide labels on panel subplots to reduce clutter)
        format_utm_axes(ax, show_labels=False)

    fig.suptitle(
        f"{area_name} — Morphometric Indicators (10m grid)",
        fontsize=14,
        fontweight="bold",
        color=SWISS["primary"],
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 3 saved: %s", output_path)


# =========================================================================
# Figure 4 -- SVF vs Slope Scatter
# =========================================================================


def figure_svf_slope_scatter(
    grid: gpd.GeoDataFrame,
    output_path: Path,
    area_name: str = "Study Area",
):
    """Figure 4: SVF vs slope scatter with marginal histograms, colored by lambda_p."""
    apply_publication_style()

    cols = ["svf", "slope_deg", "lambda_p"]
    valid = grid.dropna(subset=cols).copy()
    if len(valid) < 5:
        logger.warning("Too few valid cells for SVF-slope scatter; skipping.")
        return

    fig = plt.figure(figsize=(12, 9))

    # Main axes + marginals
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Scatter
    sc = ax_main.scatter(
        valid["slope_deg"],
        valid["svf"],
        c=valid["lambda_p"],
        cmap="YlOrRd",
        s=12,
        alpha=0.6,
        edgecolors="none",
        vmin=0,
        vmax=min(1.0, valid["lambda_p"].quantile(0.98)),
    )
    cb = fig.colorbar(sc, ax=ax_main, shrink=0.6, pad=0.02)
    cb.set_label(r"$\lambda_p$", fontsize=10)

    ax_main.set_xlabel("Terrain Slope (degrees)", fontsize=11)
    ax_main.set_ylabel("Sky View Factor", fontsize=11)
    ax_main.axhline(
        0.15, color=SWISS["accent"], linestyle="--", linewidth=0.8, alpha=0.7
    )
    ax_main.text(
        ax_main.get_xlim()[1] * 0.95,
        0.16,
        "SVF = 0.15",
        ha="right",
        fontsize=7,
        color=SWISS["accent"],
    )

    # Marginal histograms
    ax_histx.hist(
        valid["slope_deg"], bins=30, color=SWISS["accent"], alpha=0.5, edgecolor="white"
    )
    ax_histx.set_ylabel("Count", fontsize=8)
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    ax_histy.hist(
        valid["svf"],
        bins=30,
        orientation="horizontal",
        color=SWISS["accent"],
        alpha=0.5,
        edgecolor="white",
    )
    ax_histy.set_xlabel("Count", fontsize=8)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    # Correlation annotation
    from scipy.stats import pearsonr

    r, p = pearsonr(valid["slope_deg"], valid["svf"])
    ax_main.text(
        0.03,
        0.97,
        f"r = {r:.3f} (p = {p:.1e})\nn = {len(valid):,}",
        transform=ax_main.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
            edgecolor=SWISS["grid"],
        ),
    )

    fig.suptitle(
        f"{area_name} — SVF vs. Terrain Slope",
        fontsize=13,
        fontweight="bold",
        color=SWISS["primary"],
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 4 saved: %s", output_path)


# =========================================================================
# Figure 5 -- Correlation Matrix
# =========================================================================


def figure_correlation_matrix(
    grid: gpd.GeoDataFrame,
    output_path: Path,
    area_name: str = "Study Area",
):
    """Figure 5: Pairwise Pearson correlation heatmap."""
    apply_publication_style()

    indicator_cols = [
        "svf",
        "lambda_p",
        "lambda_f_mean",
        "porosity",
        "sigma_h",
        "H_mean",
        "street_orientation_entropy",
        "slope_deg",
    ]
    labels = [
        "SVF",
        r"$\lambda_p$",
        r"$\lambda_f$",
        "Porosity",
        r"$\sigma_H$",
        r"$\bar{H}$",
        "Entropy",
        "Slope",
    ]

    available = [c for c in indicator_cols if c in grid.columns]
    avail_labels = [labels[indicator_cols.index(c)] for c in available]

    if len(available) < 3:
        logger.warning("Too few indicators for correlation matrix; skipping.")
        return

    data = grid[available].dropna()
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 9))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else SWISS["primary"]
            weight = "bold" if abs(val) > 0.7 else "normal"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight=weight,
            )

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(avail_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(avail_labels, fontsize=9)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Pearson r", fontsize=10)

    # Flag highly correlated pairs
    high_corr_pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if abs(corr.values[i, j]) > 0.7:
                high_corr_pairs.append(
                    f"{avail_labels[i]}--{avail_labels[j]}: r={corr.values[i, j]:.2f}"
                )

    if high_corr_pairs:
        note = "High correlation (|r| > 0.7): " + "; ".join(high_corr_pairs)
        fig.text(
            0.5,
            0.01,
            note,
            ha="center",
            fontsize=7.5,
            color=SWISS["caption"],
            style="italic",
        )

    ax.set_title(
        f"{area_name} — Indicator Correlation Matrix",
        fontsize=13,
        fontweight="bold",
        color=SWISS["primary"],
        pad=12,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 saved: %s", output_path)


# =========================================================================
# Figure 6 -- Distributions (Violin Plots)
# =========================================================================


def figure_distributions(
    grid: gpd.GeoDataFrame,
    output_path: Path,
    area_name: str = "Study Area",
):
    """Figure 6: Violin plots of morphometric distributions with LCZ3 reference bands."""
    apply_publication_style()

    # Indicators and their LCZ3 compact low-rise reference ranges
    indicators = {
        "svf": {"label": "SVF", "lcz_range": (0.2, 0.6), "lcz_label": "LCZ 3"},
        "lambda_p": {
            "label": r"$\lambda_p$",
            "lcz_range": (0.4, 0.7),
            "lcz_label": "LCZ 3",
        },
        "lambda_f_mean": {"label": r"$\lambda_f$", "lcz_range": None},
        "porosity": {"label": "Porosity", "lcz_range": None},
        "sigma_h": {"label": r"$\sigma_H$ (m)", "lcz_range": None},
        "H_mean": {
            "label": r"$\bar{H}$ (m)",
            "lcz_range": (3, 10),
            "lcz_label": "LCZ 3",
        },
        "slope_deg": {"label": "Slope (deg)", "lcz_range": None},
    }

    available = {k: v for k, v in indicators.items() if k in grid.columns}
    if len(available) < 3:
        logger.warning("Too few indicators for distributions; skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(16, 7))
    if n == 1:
        axes = [axes]

    for idx, (col, meta) in enumerate(available.items()):
        ax = axes[idx]
        data = grid[col].dropna().values

        if len(data) == 0:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=9
            )
            ax.set_title(meta["label"])
            continue

        parts = ax.violinplot(data, positions=[0], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(SWISS["accent"])
            pc.set_alpha(0.6)
        for key in ("cmins", "cmaxes", "cmedians", "cbars"):
            if key in parts:
                parts[key].set_color(SWISS["primary"])

        # LCZ3 reference band
        lcz = meta.get("lcz_range")
        if lcz is not None:
            ax.axhspan(lcz[0], lcz[1], color="#22c55e", alpha=0.15, zorder=0)
            ax.text(
                0.95,
                lcz[1],
                meta.get("lcz_label", "LCZ 3"),
                transform=mpl.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData
                ),
                fontsize=6,
                color="#16a34a",
                ha="right",
                va="bottom",
            )

        # Stats annotation
        ax.text(
            0.95,
            0.02,
            f"n={len(data):,}\n\u03bc={np.mean(data):.2f}\n\u03c3={np.std(data):.2f}",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.8,
                edgecolor=SWISS["grid"],
            ),
        )

        ax.set_title(meta["label"], fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"{area_name} — Morphometric Distributions",
        fontsize=13,
        fontweight="bold",
        color=SWISS["primary"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 6 saved: %s", output_path)


# =========================================================================
# Figure 7 -- Street-Level SVF Map  (NEW)
# =========================================================================


def figure_street_svf_map(
    streets_svf: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    dtm_path: Path,
    output_path: Path,
    boundary: Optional[gpd.GeoDataFrame] = None,
    area_name: str = "Study Area",
):
    """Figure 7: Street-level SVF -- segments colored by svf_mean on hillshade DTM.

    Parameters
    ----------
    streets_svf : GeoDataFrame
        Street segments with columns ``svf_mean``, ``n_points``, ``geometry``.
    buildings : GeoDataFrame
        Building footprints.
    dtm_path : Path
        Path to GeoTIFF DTM raster.
    output_path : Path
        Destination for the saved figure.
    boundary : GeoDataFrame, optional
        Settlement boundary polygon.
    area_name : str
        Name shown in the title.
    """
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Hillshade background + contours
    _add_hillshade_background(ax, dtm_path)
    add_terrain_contours(ax, dtm_path)

    # Building footprints (light gray background)
    buildings.plot(
        ax=ax,
        facecolor=SWISS["building_face"],
        edgecolor=SWISS["building_edge"],
        linewidth=0.2,
        alpha=0.5,
    )

    # Boundary
    if boundary is not None:
        boundary.boundary.plot(
            ax=ax, color=SWISS["primary"], linewidth=1.2, linestyle="--", alpha=0.6
        )

    # -- Street segments colored by svf_mean --
    valid = streets_svf[streets_svf["svf_mean"].notna()].copy()
    if valid.empty:
        logger.warning("No street SVF data for map; skipping.")
        plt.close(fig)
        return

    # Linewidth proportional to n_points (higher n_points = thicker = more confident)
    if "n_points" in valid.columns:
        np_vals = valid["n_points"].values.astype(float)
        np_min, np_max = np_vals.min(), np_vals.max()
        if np_max > np_min:
            lw = 0.8 + 2.2 * (np_vals - np_min) / (np_max - np_min)
        else:
            lw = np.full_like(np_vals, 1.5)
    else:
        lw = np.full(len(valid), 1.5)

    vmin = valid["svf_mean"].quantile(0.01)
    vmax = valid["svf_mean"].quantile(0.99)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("cividis")

    # Plot each segment individually so linewidth can vary
    for i, (_, row) in enumerate(valid.iterrows()):
        color = cmap(norm(row["svf_mean"]))
        gpd.GeoDataFrame([row], geometry="geometry", crs=valid.crs).plot(
            ax=ax, color=color, linewidth=lw[i], zorder=3
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Street-Level SVF (mean)", fontsize=10)

    # Annotate darkest streets (lowest SVF)
    n_annotate = min(3, len(valid))
    darkest = valid.nsmallest(n_annotate, "svf_mean")
    for _, row in darkest.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(
            f"SVF={row['svf_mean']:.2f}",
            xy=(centroid.x, centroid.y),
            fontsize=6.5,
            color="#b91c1c",
            fontweight="bold",
            ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=5,
        )

    ax.set_title(
        f"{area_name} — Street-Level Sky View Factor",
        fontsize=14,
        fontweight="bold",
        color=SWISS["primary"],
        pad=12,
    )

    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)

    # Summary annotation
    ax.text(
        0.02,
        0.02,
        f"{len(valid):,} segments | "
        f"mean SVF = {valid['svf_mean'].mean():.3f} | "
        f"min = {valid['svf_mean'].min():.3f}",
        transform=ax.transAxes,
        fontsize=7,
        color=SWISS["secondary"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor=SWISS["grid"],
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 7 saved: %s", output_path)


# =========================================================================
# Figure 8 -- Building Heights  (NEW)
# =========================================================================


def figure_building_heights(
    buildings: gpd.GeoDataFrame,
    dtm_path: Path,
    output_path: Path,
    boundary: Optional[gpd.GeoDataFrame] = None,
    area_name: str = "Study Area",
):
    """Figure 8: Building footprints colored by height on hillshade DTM.

    Parameters
    ----------
    buildings : GeoDataFrame
        Building footprints with ``altura`` (or ``height``) column.
    dtm_path : Path
        Path to GeoTIFF DTM raster.
    output_path : Path
        Destination for the saved figure.
    boundary : GeoDataFrame, optional
        Settlement boundary polygon.
    area_name : str
        Name shown in the title.
    """
    apply_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Hillshade background + contours
    _add_hillshade_background(ax, dtm_path)
    add_terrain_contours(ax, dtm_path)

    # Boundary
    if boundary is not None:
        boundary.boundary.plot(
            ax=ax, color=SWISS["primary"], linewidth=1.2, linestyle="--", alpha=0.6
        )

    # Determine height column
    if "altura" in buildings.columns:
        height_col = "altura"
    elif "height" in buildings.columns:
        height_col = "height"
    else:
        logger.warning(
            "No height column (altura/height) in buildings; falling back to flat fill."
        )
        buildings.plot(
            ax=ax,
            facecolor=SWISS["building_face"],
            edgecolor=SWISS["building_edge"],
            linewidth=0.3,
            alpha=0.85,
        )
        ax.set_title(
            f"{area_name} — Building Heights (no data)",
            fontsize=14,
            fontweight="bold",
            color=SWISS["primary"],
            pad=12,
        )
        format_utm_axes(ax)
        add_scale_bar(ax)
        add_north_arrow(ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure 8 saved (no height data): %s", output_path)
        return

    valid_heights = buildings[height_col].replace([np.inf, -np.inf], np.nan).dropna()
    vmin = valid_heights.min() if len(valid_heights) > 0 else 0
    vmax = valid_heights.max() if len(valid_heights) > 0 else 1

    buildings.plot(
        ax=ax,
        column=height_col,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        edgecolor="#555555",
        linewidth=0.3,
        legend=True,
        legend_kwds={
            "label": "Building Height (m)",
            "shrink": 0.6,
            "pad": 0.02,
        },
        zorder=2,
    )

    ax.set_title(
        f"{area_name} — Building Heights",
        fontsize=14,
        fontweight="bold",
        color=SWISS["primary"],
        pad=12,
    )

    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)

    # Summary annotation
    mean_h = valid_heights.mean()
    max_h = valid_heights.max()
    ax.text(
        0.02,
        0.02,
        f"{len(buildings):,} buildings | "
        f"mean height = {mean_h:.1f} m | "
        f"max = {max_h:.1f} m",
        transform=ax.transAxes,
        fontsize=7,
        color=SWISS["secondary"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor=SWISS["grid"],
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 8 saved: %s", output_path)

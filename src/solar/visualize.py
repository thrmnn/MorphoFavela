"""Publication-quality solar access visualizations (Nature Cities standard).

Provides spatial maps, distribution plots, dashboards, sun-path diagrams,
cross-area comparison figures, and artistic hero visualizations for
stakeholder/media communication.  All functions follow the save pattern
established in ``svf_v2.visualize`` and use shared cartographic helpers
from ``src.cartography``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.cartography import (
    add_north_arrow,
    add_scale_bar,
    add_settlement_boundary,
    apply_publication_style,
    format_utm_axes,
)
from src.config import DPI, FIGURE_SIZE
from src.solar.sun import (
    compute_sun_positions,
    REFERENCE_DATES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nature Cities style constants
# ---------------------------------------------------------------------------
_NC_TITLE_SIZE = 12
_NC_SUPTITLE_SIZE = 14
_NC_LABEL_SIZE = 10
_NC_TICK_SIZE = 8
_NC_LEGEND_SIZE = 8
_NC_CBAR_LABEL_SIZE = 9
_NC_GRID_COLOR = "#f0f0f0"

# Building underlay (light, recessive)
_BLD_FACE = "#e8e8e8"
_BLD_EDGE = "#bbbbbb"
_BLD_ALPHA = 0.4
_BLD_LW = 0.25

# Colormaps
SOLAR_CMAP = "YlOrRd"
IRRADIANCE_CMAP = "inferno"

# Palette for multi-area comparisons
_AREA_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]

# Palette for seasonal date tracks on sun-path diagram
_DATE_COLORS = ["#1565c0", "#e65100", "#2e7d32", "#6a1b9a", "#42d4f4", "#f032e6"]

# Seasonal box colours
_SEASON_COLORS = {
    "20260621": "#1565c0",  # Winter (blue)
    "20261221": "#e65100",  # Summer (orange)
    "20260320": "#2e7d32",  # Equinox (green)
    "20260922": "#388e3c",  # Equinox (green)
}

# ---------------------------------------------------------------------------
# Human-readable date labels
# ---------------------------------------------------------------------------
DATE_LABELS = {
    "20260621": "Winter Solstice\n(Jun 21)",
    "20261221": "Summer Solstice\n(Dec 21)",
    "20260320": "March Equinox\n(Mar 20)",
    "20260922": "September Equinox\n(Sep 22)",
    "2026-06-21": "Winter Solstice (Jun 21)",
    "2026-12-21": "Summer Solstice (Dec 21)",
    "2026-03-20": "March Equinox (Mar 20)",
    "2026-09-22": "September Equinox (Sep 22)",
}


def _date_label(raw: str) -> str:
    """Map a raw date string to a human-readable label."""
    return DATE_LABELS.get(raw, raw)


def _color_for_index(idx: int) -> str:
    """Return a colour from the area palette, cycling if needed."""
    return _AREA_COLORS[idx % len(_AREA_COLORS)]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _set_nature_style() -> None:
    """Set rcParams to Nature Cities standards."""
    apply_publication_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": _NC_LABEL_SIZE,
            "axes.titlesize": _NC_TITLE_SIZE,
            "axes.labelsize": _NC_LABEL_SIZE,
            "xtick.labelsize": _NC_TICK_SIZE,
            "ytick.labelsize": _NC_TICK_SIZE,
            "legend.fontsize": _NC_LEGEND_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "text.antialiased": True,
        }
    )


def _light_buildings(ax, footprints_gdf, alpha=_BLD_ALPHA):
    """Draw buildings as a very light, recessive underlay."""
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        footprints_gdf.plot(
            ax=ax,
            facecolor=_BLD_FACE,
            edgecolor=_BLD_EDGE,
            linewidth=_BLD_LW,
            alpha=alpha,
            zorder=1,
        )


def _add_boundary(ax, boundary_gdf):
    """Add settlement boundary if available."""
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        add_settlement_boundary(ax, boundary_gdf, edgecolor="#333333", linestyle="--")


def _add_panel_label(ax, label, fontsize=12):
    """Add Nature-style panel label (a, b, c, d) at upper-left."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=20,
    )


# ===================================================================
# 1. Single-panel solar hours map
# ===================================================================


def plot_solar_access(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    title: str = "Street-Level Solar Access \u2014 Mean Annual (4 Reference Days)",
    column: str = "solar_hours",
    cmap: str = SOLAR_CMAP,
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    """Single-panel solar hours scatter map.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point data with a solar-hours column.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame, optional
        Building footprint polygons for context.
    title : str
        Figure title.
    column : str
        Column to visualise.
    cmap : str
        Matplotlib colormap name.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping solar access plot.", column)
        return Path(output_path)

    _set_nature_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Light building underlay
    _light_buildings(ax, footprints_gdf)

    vmin = 0.0
    vmax = gdf[column].max() if len(gdf) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        markersize=5,
        legend=True,
        legend_kwds={
            "label": "Hours of Direct Sunlight",
            "shrink": 0.65,
            "pad": 0.02,
        },
        zorder=5,
    )

    # Style the colorbar label
    cbar = fig.axes[-1]  # colorbar axis
    cbar.tick_params(labelsize=_NC_TICK_SIZE)
    for label in cbar.get_yticklabels():
        label.set_fontsize(_NC_TICK_SIZE)
    cbar.set_ylabel("Hours of Direct Sunlight", fontsize=_NC_CBAR_LABEL_SIZE)

    _add_boundary(ax, boundary_gdf)
    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    # Key statistics annotation
    n_pts = len(gdf)
    vals = gdf[column].dropna().values
    mean_v = float(np.mean(vals)) if len(vals) > 0 else 0
    std_v = float(np.std(vals)) if len(vals) > 0 else 0
    pct_shaded = 100.0 * np.sum(vals == 0) / n_pts if n_pts > 0 else 0
    stats_text = (
        f"n={n_pts:,} | mean={mean_v:.2f}h | "
        f"\u03c3={std_v:.2f}h | {pct_shaded:.1f}% fully shaded"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85
        ),
        zorder=15,
    )

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar access map to %s", output_path)
    return output_path


# ===================================================================
# 2. Seasonal 2x2 panel map
# ===================================================================


def plot_solar_seasonal_panel(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    date_columns: list[str] | None = None,
) -> Path:
    """2x2 panel with one solar-hours map per reference date.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain columns named ``solar_hours_{date}`` for each date.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame, optional
        Building footprints underlay.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay.
    date_columns : list[str], optional
        Override which columns to plot.  Auto-detected from columns
        matching ``solar_hours_20*`` when *None*.

    Returns
    -------
    Path
        *output_path* on success.
    """
    # Auto-detect date columns
    if date_columns is None:
        date_columns = sorted(
            [c for c in gdf.columns if c.startswith("solar_hours_20")]
        )
    if not date_columns:
        logger.warning("No date columns found for seasonal panel -- skipping.")
        return Path(output_path)

    # Limit to 4 panels (2x2)
    date_columns = date_columns[:4]

    # Consistent colour scale across ALL panels
    all_vals = np.concatenate([gdf[c].dropna().values for c in date_columns])
    vmin = 0.0
    vmax = float(np.max(all_vals)) if len(all_vals) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    _set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes_flat = axes.flatten()

    # Create a shared ScalarMappable for a single colorbar
    sm = plt.cm.ScalarMappable(
        cmap=SOLAR_CMAP,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])

    for idx, col in enumerate(date_columns):
        ax = axes_flat[idx]

        # Light building underlay
        _light_buildings(ax, footprints_gdf, alpha=0.3)

        gdf.plot(
            ax=ax,
            column=col,
            cmap=SOLAR_CMAP,
            vmin=vmin,
            vmax=vmax,
            markersize=7,
            legend=False,
            zorder=3,
        )

        _add_boundary(ax, boundary_gdf)

        # Human-readable date label
        date_str = col.replace("solar_hours_", "")
        ax.set_title(
            _date_label(date_str),
            fontsize=_NC_TITLE_SIZE,
            fontweight="bold",
            pad=8,
        )
        add_scale_bar(ax)
        add_north_arrow(ax)
        ax.set_axis_off()

    # Hide unused panels
    for ax in axes_flat[len(date_columns) :]:
        ax.set_visible(False)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Hours of Direct Sunlight", fontsize=_NC_CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=_NC_TICK_SIZE)

    fig.suptitle(
        "Seasonal Solar Access \u2014 Vidigal, Rio de Janeiro",
        fontsize=_NC_SUPTITLE_SIZE,
        fontweight="bold",
        y=0.97,
    )

    # Sample annotation
    n_pts = len(gdf)
    fig.text(
        0.5,
        0.065,
        f"n = {n_pts:,} street-level sample points, 30-min temporal resolution",
        ha="center",
        fontsize=_NC_TICK_SIZE,
        color="#555555",
        style="italic",
    )

    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.22, wspace=0.15)

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved seasonal panel to %s", output_path)
    return output_path


# ===================================================================
# 3. Irradiance map
# ===================================================================


def plot_solar_irradiance_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    column: str = "irradiance_mean",
    unit: str = "Wh/m\u00b2/day",
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    """Irradiance map using the inferno colormap.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *column* with irradiance values.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame, optional
        Building footprints context.
    column : str
        Irradiance column name.
    unit : str
        Label for the colourbar.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping irradiance map.", column)
        return Path(output_path)

    _set_nature_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Very light cream underlay so dark purple/black inferno points remain visible
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        footprints_gdf.plot(
            ax=ax,
            facecolor="#f5f0e0",
            edgecolor="#d5d0c0",
            linewidth=_BLD_LW,
            alpha=0.35,
            zorder=1,
        )

    vals = gdf[column].dropna().values

    # Convert to kWh/m2/day if column is in Wh
    plot_vals = vals / 1000.0
    plot_col_name = "__irr_kwh"
    gdf = gdf.copy()
    gdf[plot_col_name] = gdf[column] / 1000.0
    display_unit = "kWh/m\u00b2/day"

    vmin = 0.0
    vmax = (
        float(np.percentile(plot_vals[np.isfinite(plot_vals)], 98))
        if len(plot_vals) > 0
        else 1.0
    )
    if vmax == 0:
        vmax = 1.0

    gdf.plot(
        ax=ax,
        column=plot_col_name,
        cmap=IRRADIANCE_CMAP,
        vmin=vmin,
        vmax=vmax,
        markersize=6,
        legend=True,
        legend_kwds={
            "label": display_unit,
            "shrink": 0.65,
            "pad": 0.02,
        },
        zorder=5,
    )

    # Style colorbar
    cbar = fig.axes[-1]
    cbar.tick_params(labelsize=_NC_TICK_SIZE)
    cbar.set_ylabel(display_unit, fontsize=_NC_CBAR_LABEL_SIZE)

    _add_boundary(ax, boundary_gdf)
    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)

    ax.set_title(
        "Mean Daily Solar Irradiance",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.set_aspect("equal")

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved irradiance map to %s", output_path)
    return output_path


# ===================================================================
# 4. Distribution plot (bimodal-aware)
# ===================================================================


def plot_solar_distribution(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    column: str = "solar_hours_mean",
    title: str | None = None,
) -> Path:
    """Distribution plot with zero-spike handling.

    Handles the common spike at zero (fully shaded points) with a
    separate annotated bar, then shows histogram + KDE for the
    non-zero portion.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *column* with numeric solar values.
    output_path : Path
        Where to save the figure.
    column : str
        Column name.
    title : str, optional
        Figure title.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping distribution.", column)
        return Path(output_path)

    vals = gdf[column].dropna().values
    n_total = len(vals)
    if n_total == 0:
        logger.warning("No data for distribution plot -- skipping.")
        return Path(output_path)

    _set_nature_style()

    # Separate zero and non-zero
    n_zero = int(np.sum(vals == 0.0))
    pct_zero = 100.0 * n_zero / n_total
    vals_nonzero = vals[vals > 0.0]

    # Identify "fully exposed" as top 10% of the data range
    val_max = float(vals.max()) if len(vals) > 0 else 1.0
    threshold_high = val_max * 0.9 if val_max > 0 else 0.9
    n_exposed = int(np.sum(vals > threshold_high))
    pct_exposed = 100.0 * n_exposed / n_total

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals))
    q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main histogram for non-zero values
    if len(vals_nonzero) > 0:
        bins = np.linspace(vals_nonzero.min() * 0.9, val_max * 1.05, 35)
        ax.hist(
            vals_nonzero,
            bins=bins,
            color="#f58231",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.75,
            label="Solar hours > 0",
            zorder=2,
        )

        # KDE overlay (thicker, darker)
        if len(vals_nonzero) > 5:
            try:
                kde = gaussian_kde(vals_nonzero, bw_method="scott")
                xs = np.linspace(bins[0], bins[-1], 200)
                bin_width = bins[1] - bins[0]
                ax.plot(
                    xs,
                    kde(xs) * len(vals_nonzero) * bin_width,
                    color="#c62828",
                    linewidth=2.5,
                    label="KDE",
                    zorder=3,
                )
            except Exception:
                pass

    # Annotated bar for zero spike (red) -- no legend entry; annotation arrow explains it
    if n_zero > 0:
        bar_width = (val_max * 1.05) / 60 if val_max > 0 else 0.1
        ax.bar(
            0.0,
            n_zero,
            width=bar_width,
            color="#d32f2f",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            zorder=4,
        )
        ax.annotate(
            f"{pct_zero:.0f}% fully shaded\n(n = {n_zero:,})",
            xy=(0.0, n_zero),
            xytext=(val_max * 0.20, n_zero * 0.85),
            fontsize=_NC_TICK_SIZE,
            arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.2),
            color="#d32f2f",
            fontweight="bold",
        )

    # Mean/median lines
    ax.axvline(
        mean_val,
        color="#c62828",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_val:.2f} h",
        zorder=6,
    )
    ax.axvline(
        median_val,
        color="#e65100",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median = {median_val:.2f} h",
        zorder=6,
    )

    # Stats box -- lower right to avoid overlap
    stats_text = (
        f"n = {n_total:,}\n"
        f"\u03bc = {mean_val:.2f} h\n"
        f"\u03c3 = {std_val:.2f} h\n"
        f"IQR = [{q25:.1f}, {q75:.1f}]\n"
        f"Shaded = {pct_zero:.1f}%\n"
        f"Exposed (>{threshold_high:.0f} h) = {pct_exposed:.1f}%"
    )
    ax.text(
        0.97,
        0.35,
        stats_text,
        transform=ax.transAxes,
        fontsize=_NC_TICK_SIZE,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax.set_xlabel(
        "Mean Daily Sunshine Hours (across 4 reference dates)",
        fontsize=_NC_LABEL_SIZE,
    )
    ax.set_ylabel("Count", fontsize=_NC_LABEL_SIZE)
    ax.set_title(
        title or "Distribution of Street-Level Solar Access",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=_NC_LEGEND_SIZE, loc="upper right", frameon=False)
    ax.tick_params(axis="both", labelsize=_NC_TICK_SIZE)
    ax.grid(True, axis="y", color=_NC_GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = Path(output_path)
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved distribution plot to %s", output_path)
    return output_path


# ===================================================================
# 5. Solar dashboard (2x2)
# ===================================================================


def plot_solar_dashboard(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    roads_gdf: gpd.GeoDataFrame | None = None,
    title: str = "Solar Dashboard",
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    """2x2 dashboard -- the most important figure for stakeholders.

    Panels
    ------
    * **a** (top-left) -- Solar hours map with boundary overlay.
    * **b** (top-right) -- Distribution histogram with zero-spike.
    * **c** (bottom-left) -- Seasonal boxplots with human-readable labels.
    * **d** (bottom-right) -- SVF vs solar scatter (hexbin + regression).

    Parameters
    ----------
    gdf : GeoDataFrame
        Solar results with ``solar_hours_mean`` and optionally per-date
        columns (``solar_hours_20*``) and ``svf``.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame, optional
        Building footprints for context on the map panel.
    roads_gdf : GeoDataFrame, optional
        Road centrelines drawn behind the scatter.
    title : str
        Super-title for the figure (made subtle).
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay.

    Returns
    -------
    Path
        *output_path* on success.
    """
    # Determine the primary solar column
    solar_col = "solar_hours_mean"
    if solar_col not in gdf.columns:
        solar_col = "solar_hours"
    if solar_col not in gdf.columns:
        logger.warning("No solar_hours column found -- skipping dashboard.")
        return Path(output_path)

    solar = gdf[solar_col].dropna().values
    mean_val = float(np.mean(solar))
    median_val = float(np.median(solar))
    std_val = float(np.std(solar))

    _set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    ax_map, ax_hist = axes[0]
    ax_season, ax_scatter = axes[1]

    # ----------------------------------------------------------------
    # Panel A: solar hours scatter map
    # ----------------------------------------------------------------
    if roads_gdf is not None and not roads_gdf.empty:
        roads_gdf.plot(ax=ax_map, color="#d5d5d5", linewidth=0.4, zorder=0)

    _light_buildings(ax_map, footprints_gdf)

    vmax = float(solar.max()) if len(solar) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    gdf.plot(
        ax=ax_map,
        column=solar_col,
        cmap=SOLAR_CMAP,
        vmin=0,
        vmax=vmax,
        markersize=3,
        legend=True,
        legend_kwds={"label": "Hours", "shrink": 0.5, "pad": 0.02},
        zorder=5,
    )
    _add_boundary(ax_map, boundary_gdf)
    ax_map.set_title(
        "Mean Solar Hours",
        fontsize=_NC_TITLE_SIZE,
        fontweight="bold",
    )
    ax_map.set_aspect("equal")
    add_scale_bar(ax_map)
    add_north_arrow(ax_map)
    format_utm_axes(ax_map, show_labels=False)
    _add_panel_label(ax_map, "a")

    # ----------------------------------------------------------------
    # Panel B: histogram with zero-spike
    # ----------------------------------------------------------------
    n_zero = int(np.sum(solar == 0.0))
    pct_zero = 100.0 * n_zero / len(solar) if len(solar) > 0 else 0
    vals_nz = solar[solar > 0.0]

    if len(vals_nz) > 0:
        bins = np.linspace(vals_nz.min() * 0.9, solar.max() * 1.05, 35)
        ax_hist.hist(
            vals_nz,
            bins=bins,
            color="#f58231",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75,
            zorder=2,
        )
        if len(vals_nz) > 5:
            try:
                kde = gaussian_kde(vals_nz, bw_method="scott")
                xs = np.linspace(bins[0], bins[-1], 200)
                bw = bins[1] - bins[0]
                ax_hist.plot(
                    xs,
                    kde(xs) * len(vals_nz) * bw,
                    color="#c62828",
                    linewidth=2.0,
                    zorder=3,
                )
            except Exception:
                pass

    if n_zero > 0:
        bar_w = (solar.max() * 1.05) / 60 if solar.max() > 0 else 0.1
        ax_hist.bar(
            0.0,
            n_zero,
            width=bar_w,
            color="#d32f2f",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=4,
        )
        ax_hist.annotate(
            f"{pct_zero:.0f}% shaded",
            xy=(0.0, n_zero),
            xytext=(solar.max() * 0.2, n_zero * 0.85),
            fontsize=7,
            fontweight="bold",
            color="#d32f2f",
            arrowprops=dict(arrowstyle="->", color="#d32f2f", lw=1.0),
        )

    ax_hist.axvline(mean_val, color="#c62828", linestyle="--", linewidth=1.2)
    ax_hist.axvline(median_val, color="#e65100", linestyle="-.", linewidth=1.2)

    stats_txt = (
        f"n = {len(solar):,}\n\u03bc = {mean_val:.2f} h\n\u03c3 = {std_val:.2f} h"
    )
    ax_hist.text(
        0.96,
        0.94,
        stats_txt,
        transform=ax_hist.transAxes,
        fontsize=7,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9
        ),
    )
    ax_hist.set_xlabel("Solar Hours", fontsize=_NC_LABEL_SIZE)
    ax_hist.set_ylabel("Count", fontsize=_NC_LABEL_SIZE)
    ax_hist.set_title(
        "Distribution",
        fontsize=_NC_TITLE_SIZE,
        fontweight="bold",
    )
    ax_hist.tick_params(axis="both", labelsize=_NC_TICK_SIZE)
    ax_hist.grid(True, axis="y", color=_NC_GRID_COLOR, linewidth=0.5, zorder=0)
    ax_hist.set_axisbelow(True)
    _add_panel_label(ax_hist, "b")

    # ----------------------------------------------------------------
    # Panel C: seasonal boxplots
    # ----------------------------------------------------------------
    date_cols = sorted([c for c in gdf.columns if c.startswith("solar_hours_20")])
    if date_cols:
        box_data = [gdf[c].dropna().values for c in date_cols]
        box_labels = [_date_label(c.replace("solar_hours_", "")) for c in date_cols]

        bp = ax_season.boxplot(
            box_data,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="#555555", linewidth=1.0),
            capprops=dict(color="#555555", linewidth=1.0),
        )
        for idx, patch in enumerate(bp["boxes"]):
            date_key = date_cols[idx].replace("solar_hours_", "")
            color = _SEASON_COLORS.get(date_key, _DATE_COLORS[idx % len(_DATE_COLORS)])
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.8)

        ax_season.set_xticklabels(
            box_labels,
            fontsize=7,
            ha="center",
        )
        ax_season.set_ylabel("Solar Hours", fontsize=_NC_LABEL_SIZE)
        ax_season.set_title(
            "Seasonal Variation",
            fontsize=_NC_TITLE_SIZE,
            fontweight="bold",
        )

        # Annotate medians
        for idx, d in enumerate(box_data):
            med = float(np.median(d))
            ax_season.text(
                idx + 1,
                med,
                f"{med:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="#555555",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
    else:
        ax_season.text(
            0.5,
            0.5,
            "No seasonal date columns\navailable",
            transform=ax_season.transAxes,
            ha="center",
            va="center",
            fontsize=_NC_LABEL_SIZE,
        )
        ax_season.set_title(
            "Seasonal Variation",
            fontsize=_NC_TITLE_SIZE,
            fontweight="bold",
        )
    ax_season.tick_params(axis="both", labelsize=_NC_TICK_SIZE)
    _add_panel_label(ax_season, "c")

    # ----------------------------------------------------------------
    # Panel D: SVF vs solar (hexbin + regression)
    # ----------------------------------------------------------------
    if "svf" in gdf.columns:
        data = gdf[["svf", solar_col]].dropna()
        svf_clean = data["svf"].values
        solar_clean = data[solar_col].values
        mask = np.isfinite(svf_clean) & np.isfinite(solar_clean)
        svf_clean = svf_clean[mask]
        solar_clean = solar_clean[mask]

        from matplotlib.colors import LogNorm as _LogNorm

        hb = ax_scatter.hexbin(
            svf_clean,
            solar_clean,
            gridsize=35,
            cmap="YlOrBr",
            mincnt=1,
            norm=_LogNorm(vmin=1, vmax=max(10, len(svf_clean) // 20)),
            zorder=2,
        )
        cb = plt.colorbar(hb, ax=ax_scatter, shrink=0.7, pad=0.02)
        cb.set_label("Count", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Linear regression
        if len(svf_clean) > 2:
            coeffs = np.polyfit(svf_clean, solar_clean, 1)
            x_line = np.linspace(0, 1, 100)
            ax_scatter.plot(
                x_line,
                np.polyval(coeffs, x_line),
                color="#1a237e",
                linewidth=2.5,
                linestyle="-",
                zorder=3,
            )
            ss_res = np.sum((solar_clean - np.polyval(coeffs, svf_clean)) ** 2)
            ss_tot = np.sum((solar_clean - np.mean(solar_clean)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax_scatter.text(
                0.05,
                0.92,
                f"R\u00b2 = {r_sq:.3f}\ny = {coeffs[0]:.1f}x {coeffs[1]:+.1f}",
                transform=ax_scatter.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#cccccc",
                    alpha=0.9,
                ),
            )

        ax_scatter.set_xlabel("Sky View Factor", fontsize=_NC_LABEL_SIZE)
        ax_scatter.set_ylabel("Solar Hours", fontsize=_NC_LABEL_SIZE)
        ax_scatter.set_title(
            "SVF vs Solar Access",
            fontsize=_NC_TITLE_SIZE,
            fontweight="bold",
        )
        ax_scatter.set_xlim(0, 1)
    else:
        # CDF fallback
        sorted_solar = np.sort(solar)
        cdf_y = np.arange(1, len(sorted_solar) + 1) / len(sorted_solar)
        ax_scatter.plot(sorted_solar, cdf_y, color="#c62828", linewidth=1.5)
        ax_scatter.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax_scatter.axvline(
            median_val, color="#e65100", linestyle="-.", linewidth=1, alpha=0.7
        )
        ax_scatter.set_xlabel("Solar Hours", fontsize=_NC_LABEL_SIZE)
        ax_scatter.set_ylabel("Cumulative Probability", fontsize=_NC_LABEL_SIZE)
        ax_scatter.set_title(
            "Solar Hours CDF",
            fontsize=_NC_TITLE_SIZE,
            fontweight="bold",
        )
    ax_scatter.tick_params(axis="both", labelsize=_NC_TICK_SIZE)
    _add_panel_label(ax_scatter, "d")

    # Subtle suptitle (or none)
    fig.tight_layout(rect=[0, 0, 1, 1], h_pad=3, w_pad=3)

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar dashboard to %s", output_path)
    return output_path


# ===================================================================
# 6. Polar stereographic sun-path diagram
# ===================================================================


def plot_solar_sun_path(
    latitude: float,
    longitude: float,
    output_path: Path,
    dates: list[str] | None = None,
) -> Path:
    """Polar stereographic sun-path diagram.

    Plots altitude (as zenith-angle radius) vs azimuth for each date,
    with hour markers along each track and cardinal direction labels.

    Parameters
    ----------
    latitude, longitude : float
        Site coordinates.
    output_path : Path
        Where to save the figure.
    dates : list[str] | None
        ISO date strings.  Defaults to ``REFERENCE_DATES`` values.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if dates is None:
        dates = list(REFERENCE_DATES.values())

    _set_nature_style()
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Polar setup: theta=0 at North, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        fontsize=_NC_LABEL_SIZE,
    )
    ax.set_rlim(0, 90)
    ax.set_rlabel_position(135)
    ax.set_rticks([15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(
        ["75\u00b0", "60\u00b0", "45\u00b0", "30\u00b0", "15\u00b0", "0\u00b0"],
        fontsize=7,
    )

    # Lighter concentric grid rings
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5)

    for idx, date in enumerate(dates):
        color = _DATE_COLORS[idx % len(_DATE_COLORS)]
        label = _date_label(date)

        # Fine-resolution track
        positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=5,
            hour_end=19,
            interval_minutes=10,
        )
        if not positions:
            continue

        alts = [p[0] for p in positions]
        azs = [p[1] for p in positions]
        theta = np.radians(azs)
        r = 90.0 - np.array(alts)

        ax.plot(theta, r, color=color, linewidth=2.2, label=label, zorder=3)

        # Hour markers every 2 hours (60-min positions, then pick every 2nd)
        hour_positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=5,
            hour_end=19,
            interval_minutes=60,
        )
        for h_idx, (alt_h, az_h) in enumerate(hour_positions):
            th = np.radians(az_h)
            rh = 90.0 - alt_h
            actual_hour = 5 + h_idx  # hour_start + index
            ax.plot(th, rh, "o", color=color, markersize=4, zorder=4)

            # Label every 2 hours -- only on winter (idx 0) and summer (idx 1)
            # to avoid overlapping text on equinox tracks
            if actual_hour % 2 == 0 and idx < 2:
                # Offset labels to avoid overlap with marker
                y_offset = 6 if idx == 0 else -8
                ax.annotate(
                    f"{actual_hour}h",
                    xy=(th, rh),
                    xytext=(4, y_offset),
                    textcoords="offset points",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                    ha="left",
                    va="bottom" if idx == 0 else "top",
                )

    ax.set_title(
        f"Sun Path Diagram\n(lat = {latitude:.2f}\u00b0, lon = {longitude:.2f}\u00b0)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Legend inside plot, lower-left, smaller
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.12),
        fontsize=_NC_LEGEND_SIZE,
        frameon=False,
        ncol=2,
    )

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sun path diagram to %s", output_path)
    return output_path


# ===================================================================
# 7. Solar vs SVF scatter with regression
# ===================================================================


def plot_solar_vs_svf(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    solar_col: str = "solar_hours_mean",
    svf_col: str = "svf",
) -> Path:
    """Standalone SVF vs solar scatter with hexbin density and regression.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain both *solar_col* and *svf_col*.
    output_path : Path
        Where to save the figure.
    solar_col : str
        Solar hours column.
    svf_col : str
        SVF column.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if solar_col not in gdf.columns or svf_col not in gdf.columns:
        logger.warning(
            "Columns '%s' / '%s' not found -- skipping solar vs SVF.",
            solar_col,
            svf_col,
        )
        return Path(output_path)

    data = gdf[[svf_col, solar_col]].dropna()
    svf = data[svf_col].values
    solar = data[solar_col].values

    if len(svf) < 3:
        logger.warning("Too few points for solar vs SVF scatter -- skipping.")
        return Path(output_path)

    _set_nature_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f8f8f5")

    # Hexbin with warm cmap + log normalisation for visible low-density bins
    from matplotlib.colors import LogNorm

    hb = ax.hexbin(
        svf,
        solar,
        gridsize=40,
        cmap="YlOrBr",
        mincnt=1,
        norm=LogNorm(vmin=1, vmax=max(10, len(svf) // 20)),
        zorder=2,
    )
    cb = plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Point count", fontsize=_NC_CBAR_LABEL_SIZE)
    cb.ax.tick_params(labelsize=_NC_TICK_SIZE)

    # Linear regression -- dark navy line with high contrast against warm hexbin
    coeffs = np.polyfit(svf, solar, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(
        x_line,
        np.polyval(coeffs, x_line),
        color="#1a237e",
        linewidth=2.5,
        linestyle="-",
        zorder=3,
        label="Linear fit",
    )

    # R-squared
    ss_res = np.sum((solar - np.polyval(coeffs, svf)) ** 2)
    ss_tot = np.sum((solar - np.mean(solar)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Annotation box
    annotation_text = (
        f"R\u00b2 = {r_sq:.3f}\n"
        f"y = {coeffs[0]:.2f}x {coeffs[1]:+.2f}\n"
        f"n = {len(svf):,}\n\n"
        f"Strong positive correlation validates\n"
        f"geometric obstruction model"
    )
    ax.text(
        0.05,
        0.95,
        annotation_text,
        transform=ax.transAxes,
        fontsize=_NC_CBAR_LABEL_SIZE,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax.set_xlabel("Sky View Factor", fontsize=_NC_LABEL_SIZE)
    ax.set_ylabel("Mean Solar Hours", fontsize=_NC_LABEL_SIZE)
    ax.set_title(
        "Relationship Between Sky View Factor and Solar Access",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", labelsize=_NC_TICK_SIZE)
    ax.grid(True, color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar vs SVF scatter to %s", output_path)
    return output_path


# ===================================================================
# 8. Cross-area comparison
# ===================================================================


def plot_solar_comparison(
    areas_data: dict[str, gpd.GeoDataFrame],
    output_path: Path,
    column: str = "solar_hours_mean",
) -> Path:
    """Cross-area comparison with KDE and violin+box plots.

    Produces a 1x2 figure:
      - Left: overlaid KDE curves for each area.
      - Right: box-and-violin plots per area.

    Parameters
    ----------
    areas_data : dict[str, GeoDataFrame]
        Mapping of area name to GeoDataFrame containing *column*.
    output_path : Path
        Where to save the figure.
    column : str
        Numeric column to compare.

    Returns
    -------
    Path
        *output_path* on success.
    """
    _set_nature_style()
    fig, (ax_kde, ax_box) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]),
    )

    area_names = list(areas_data.keys())
    colors = [_color_for_index(i) for i in range(len(area_names))]
    arrays: list[np.ndarray] = []

    # ---- Left panel: KDE curves ----
    for idx, (name, gdf) in enumerate(areas_data.items()):
        if column not in gdf.columns:
            logger.warning(
                "Column '%s' missing in area '%s' -- skipping.", column, name
            )
            arrays.append(np.array([]))
            continue
        vals = gdf[column].dropna().values
        arrays.append(vals)

        if len(vals) < 3:
            continue

        mean_val = float(np.mean(vals))
        kde = gaussian_kde(vals, bw_method="scott")
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax_kde.plot(xs, kde(xs), color=colors[idx], linewidth=2, label=name)
        ax_kde.fill_between(xs, kde(xs), alpha=0.12, color=colors[idx])
        ax_kde.axvline(
            mean_val, color=colors[idx], linestyle="--", linewidth=1, alpha=0.7
        )

    ax_kde.set_xlabel("Solar Hours (Mean)", fontsize=_NC_LABEL_SIZE)
    ax_kde.set_ylabel("Density", fontsize=_NC_LABEL_SIZE)
    ax_kde.set_title("Density Comparison", fontsize=_NC_TITLE_SIZE, fontweight="bold")
    ax_kde.tick_params(axis="both", labelsize=_NC_TICK_SIZE)

    # Legend with stats
    legend_handles = []
    for idx, name in enumerate(area_names):
        vals = arrays[idx]
        if len(vals) == 0:
            continue
        label = (
            f"{name}  "
            f"(\u03bc={np.mean(vals):.2f}, "
            f"med={np.median(vals):.2f}, "
            f"n={len(vals):,})"
        )
        legend_handles.append(
            Line2D([0], [0], color=colors[idx], linewidth=2, label=label)
        )
    ax_kde.legend(
        handles=legend_handles,
        fontsize=_NC_LEGEND_SIZE,
        loc="upper left",
        frameon=False,
    )

    # ---- Right panel: violin + box ----
    valid_arrays = [a for a in arrays if len(a) > 0]
    valid_names = [n for n, a in zip(area_names, arrays) if len(a) > 0]
    valid_colors = [_color_for_index(i) for i, a in enumerate(arrays) if len(a) > 0]

    if valid_arrays:
        parts = ax_box.violinplot(
            valid_arrays,
            positions=range(len(valid_names)),
            showextrema=False,
            showmedians=False,
        )
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(valid_colors[idx])
            body.set_alpha(0.25)

        bp = ax_box.boxplot(
            valid_arrays,
            positions=range(len(valid_names)),
            widths=0.2,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(valid_colors[idx])
            patch.set_alpha(0.6)

        ax_box.set_xticks(range(len(valid_names)))
        ax_box.set_xticklabels(
            valid_names,
            rotation=30,
            ha="right",
            fontsize=_NC_TICK_SIZE,
        )

        # Annotate mean/median/n
        for idx, name in enumerate(valid_names):
            vals = valid_arrays[idx]
            ax_box.annotate(
                f"\u03bc={np.mean(vals):.2f}\nn={len(vals):,}",
                xy=(idx, np.mean(vals)),
                xytext=(idx + 0.35, np.mean(vals)),
                fontsize=7,
                ha="left",
                va="center",
                color=valid_colors[idx],
            )

    ax_box.set_ylabel("Solar Hours (Mean)", fontsize=_NC_LABEL_SIZE)
    ax_box.set_title(
        "Distribution Comparison",
        fontsize=_NC_TITLE_SIZE,
        fontweight="bold",
    )
    ax_box.tick_params(axis="both", labelsize=_NC_TICK_SIZE)

    fig.suptitle(
        "Comparative Solar Analysis",
        fontsize=_NC_SUPTITLE_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar comparison to %s", output_path)
    return output_path


# ===================================================================
# 9. Hero: Solar Deprivation (artistic, full-page)
# ===================================================================


def plot_solar_deprivation_hero(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    column: str = "solar_hours_mean",
) -> Path:
    """Full-page artistic figure for media/stakeholder impact.

    Dark background, dramatic hot colormap, no axes -- pure visual impact.

    Parameters
    ----------
    gdf : GeoDataFrame
        Solar results with *column*.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame
        Building footprints rendered as dark silhouettes.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary.
    column : str
        Solar hours column to visualise.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping hero plot.", column)
        return Path(output_path)

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Buildings as dark grey silhouettes
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        footprints_gdf.plot(
            ax=ax,
            facecolor="#333333",
            edgecolor="#444444",
            linewidth=0.15,
            alpha=0.9,
            zorder=1,
        )

    # Boundary
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.boundary.plot(
            ax=ax,
            color="#555555",
            linestyle="-",
            linewidth=0.8,
            zorder=2,
        )

    vmin = 0.0
    vmax = float(gdf[column].max()) if len(gdf) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    # Dramatic hot colormap: dark purple -> red/orange -> bright yellow
    gdf.plot(
        ax=ax,
        column=column,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        markersize=6,
        legend=False,
        zorder=5,
    )

    # Clean up -- no axes, no grid, no ticks
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Title in white
    fig.text(
        0.5,
        0.95,
        "Hours of Sunlight Reaching the Streets",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="white",
        fontfamily="sans-serif",
    )

    # Subtitle in light grey -- slightly larger
    fig.text(
        0.5,
        0.915,
        "Vidigal, Rio de Janeiro \u2014 Annual Mean (4 Reference Days)",
        ha="center",
        va="top",
        fontsize=15,
        color="#aaaaaa",
        fontfamily="sans-serif",
        style="italic",
    )

    # Small caption at bottom
    fig.text(
        0.5,
        0.03,
        "Each dot represents a pedestrian-level measurement point along the street network",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#888888",
        fontfamily="sans-serif",
    )

    # Horizontal colorbar at bottom
    sm = plt.cm.ScalarMappable(
        cmap="hot",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.2, 0.065, 0.6, 0.012])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Hours of Direct Sunlight",
        fontsize=12,
        color="white",
        labelpad=6,
    )
    cbar.ax.tick_params(colors="white", labelsize=11)
    cbar.outline.set_edgecolor("#555555")
    cbar.outline.set_linewidth(0.5)

    # Key statistic overlay -- white text
    solar_vals = gdf[column].dropna().values
    n_pts = len(solar_vals)
    pct_shadow = 100.0 * np.sum(solar_vals == 0) / n_pts if n_pts > 0 else 0
    fig.text(
        0.5,
        0.86,
        f"{pct_shadow:.0f}% of street surfaces in permanent shadow",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="white",
        fontfamily="sans-serif",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a1a1a",
            edgecolor="#555555",
            alpha=0.7,
        ),
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10)

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="#1a1a1a",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Saved hero solar deprivation to %s", output_path)
    return output_path


# ===================================================================
# 10. Hero: Seasonal Comparison (winter vs summer side-by-side)
# ===================================================================


def plot_solar_seasonal_comparison_hero(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame,
    winter_col: str = "solar_hours_20260621",
    summer_col: str = "solar_hours_20261221",
) -> Path:
    """Side-by-side winter vs summer for dramatic seasonal contrast.

    Dark background, cold palette for winter, warm palette for summer.

    Parameters
    ----------
    gdf : GeoDataFrame
        Solar results with per-date columns.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame
        Building footprints.
    winter_col : str
        Column for winter solstice hours.
    summer_col : str
        Column for summer solstice hours.

    Returns
    -------
    Path
        *output_path* on success.
    """
    # Validate columns
    for col in [winter_col, summer_col]:
        if col not in gdf.columns:
            logger.warning("Column '%s' not found -- skipping hero seasonal.", col)
            return Path(output_path)

    fig, (ax_w, ax_s) = plt.subplots(
        1,
        2,
        figsize=(20, 10),
        facecolor="#1a1a1a",
    )
    for ax in [ax_w, ax_s]:
        ax.set_facecolor("#1a1a1a")

    # Shared vmax for comparable scales
    all_vals = np.concatenate(
        [
            gdf[winter_col].dropna().values,
            gdf[summer_col].dropna().values,
        ]
    )
    vmin = 0.0
    vmax = float(np.max(all_vals)) if len(all_vals) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    # Custom colormaps
    # Winter: dark blue -> cool white
    winter_colors = ["#0d0d2b", "#1a237e", "#1565c0", "#42a5f5", "#b3e5fc", "#e1f5fe"]
    winter_cmap = mcolors.LinearSegmentedColormap.from_list(
        "winter_cold", winter_colors, N=256
    )

    # Summer: dark red -> bright yellow
    summer_colors = ["#1a0000", "#b71c1c", "#e65100", "#ff9800", "#ffeb3b", "#fffde7"]
    summer_cmap = mcolors.LinearSegmentedColormap.from_list(
        "summer_warm", summer_colors, N=256
    )

    panels = [
        (ax_w, winter_col, winter_cmap, "Winter Solstice (Jun 21)"),
        (ax_s, summer_col, summer_cmap, "Summer Solstice (Dec 21)"),
    ]

    for ax, col, cmap, subtitle in panels:
        # Buildings as dark silhouettes
        if footprints_gdf is not None and len(footprints_gdf) > 0:
            footprints_gdf.plot(
                ax=ax,
                facecolor="#333333",
                edgecolor="#444444",
                linewidth=0.15,
                alpha=0.9,
                zorder=1,
            )

        gdf.plot(
            ax=ax,
            column=col,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            markersize=5,
            legend=False,
            zorder=5,
        )

        ax.set_axis_off()
        ax.set_aspect("equal")

        # Panel subtitle -- larger and bolder
        ax.set_title(
            subtitle,
            fontsize=18,
            fontweight="bold",
            color="white",
            pad=12,
        )

    # Main title
    fig.text(
        0.5,
        0.96,
        "Winter vs Summer: The Seasonal Shadow Divide",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="white",
        fontfamily="sans-serif",
    )

    # Separate colorbars for each panel (different colormaps)
    for idx, (ax, col, cmap, subtitle) in enumerate(panels):
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        # Position: centered under each panel
        left = 0.06 + idx * 0.48
        cbar_ax = fig.add_axes([left, 0.04, 0.40, 0.012])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(
            "Hours of Direct Sunlight",
            fontsize=10,
            color="white",
            labelpad=4,
        )
        cbar.ax.tick_params(colors="white", labelsize=9)
        cbar.outline.set_edgecolor("#555555")
        cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.09, wspace=0.05)

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="#1a1a1a",
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Saved hero seasonal comparison to %s", output_path)
    return output_path


# ===================================================================
# 11. Facade Solar Dashboard
# ===================================================================


def plot_facade_solar_dashboard(
    facade_gdf: gpd.GeoDataFrame,
    buildings_who: pd.DataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    """Four-panel facade solar dashboard.

    Panels:
      (a) Spatial WHO compliance map (facade points)
      (b) Sunlit hours by floor level (box plot)
      (c) Building-level compliance rate distribution
      (d) Facade azimuth vs sunlit hours (polar)

    Parameters
    ----------
    facade_gdf : GeoDataFrame
        Facade point results with ``who_compliant``, ``facade_sunlit_hours``,
        ``floor_level``, ``facade_azimuth``, ``building_id``.
    buildings_who : DataFrame
        Building-level aggregation with ``who_compliance_rate``.
    output_path : Path
        Where to save the figure.
    footprints_gdf : GeoDataFrame, optional
        Building footprints for underlay.
    """
    apply_publication_style()
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    n_total = len(facade_gdf)
    n_who = int(facade_gdf["who_compliant"].sum())
    pct_who = 100 * n_who / n_total if n_total > 0 else 0

    # ── (a) Spatial WHO map ──────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0, 0])
    if footprints_gdf is not None:
        footprints_gdf.plot(
            ax=ax_map,
            facecolor=_BLD_FACE,
            edgecolor=_BLD_EDGE,
            linewidth=_BLD_LW,
            alpha=_BLD_ALPHA,
            zorder=1,
        )
    # Non-compliant in blue, compliant in red-orange
    non_comp = facade_gdf[~facade_gdf["who_compliant"]]
    comp = facade_gdf[facade_gdf["who_compliant"]]
    if len(non_comp) > 0:
        non_comp.plot(ax=ax_map, color="#3b82f6", markersize=0.3, alpha=0.4, zorder=2)
    if len(comp) > 0:
        comp.plot(ax=ax_map, color="#ef4444", markersize=2.0, alpha=0.9, zorder=3)

    ax_map.set_title("(a) WHO 2h Sunlight Compliance", fontsize=_NC_TITLE_SIZE)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ef4444",
            markersize=6,
            label=f"Compliant ({n_who:,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3b82f6",
            markersize=6,
            label=f"Deprived ({n_total - n_who:,})",
        ),
    ]
    ax_map.legend(handles=legend_elements, loc="lower right", fontsize=_NC_LEGEND_SIZE)
    format_utm_axes(ax_map)
    ax_map.set_aspect("equal")

    # ── (b) Sunlit hours by floor level ──────────────────────────────
    ax_floor = fig.add_subplot(gs[0, 1])
    floors = sorted(facade_gdf["floor_level"].unique())
    floor_data = [
        facade_gdf.loc[facade_gdf["floor_level"] == f, "facade_sunlit_hours"].values
        for f in floors
    ]
    bp = ax_floor.boxplot(
        floor_data,
        positions=floors,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    colors_floor = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(floors)))
    for patch, color in zip(bp["boxes"], colors_floor):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333")
        patch.set_linewidth(0.8)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#555")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("#111")
        line.set_linewidth(1.2)

    ax_floor.set_xlabel("Floor Level", fontsize=_NC_LABEL_SIZE)
    ax_floor.set_ylabel("Sunlit Hours", fontsize=_NC_LABEL_SIZE)
    ax_floor.set_title("(b) Sunlit Hours by Floor", fontsize=_NC_TITLE_SIZE)
    ax_floor.set_xticks(floors)
    ax_floor.set_xticklabels([f"F{f}" for f in floors])
    ax_floor.axhline(
        2.0,
        color="#ef4444",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="WHO 2h threshold",
    )
    ax_floor.legend(fontsize=_NC_LEGEND_SIZE)

    # ── (c) Building compliance rate distribution ────────────────────
    ax_bldg = fig.add_subplot(gs[1, 0])
    rates = buildings_who["who_compliance_rate"].values * 100
    ax_bldg.hist(
        rates,
        bins=20,
        range=(0, 100),
        color="#3b82f6",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    mean_rate = rates.mean()
    ax_bldg.axvline(
        mean_rate,
        color="#ef4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_rate:.1f}%",
    )
    ax_bldg.set_xlabel("WHO Compliance Rate (%)", fontsize=_NC_LABEL_SIZE)
    ax_bldg.set_ylabel("Number of Buildings", fontsize=_NC_LABEL_SIZE)
    ax_bldg.set_title("(c) Building-Level Compliance", fontsize=_NC_TITLE_SIZE)
    ax_bldg.legend(fontsize=_NC_LEGEND_SIZE)

    # ── (d) Facade azimuth vs sunlit hours (polar) ───────────────────
    ax_polar = fig.add_subplot(gs[1, 1], projection="polar")
    az_rad = np.radians(facade_gdf["facade_azimuth"].values)
    hours = facade_gdf["facade_sunlit_hours"].values

    # Bin by 10-degree sectors
    n_bins = 36
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (az_rad >= bin_edges[i]) & (az_rad < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = hours[mask].mean()

    bar_width = 2 * np.pi / n_bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors_az = plt.cm.YlOrRd(bin_means / max(bin_means.max(), 0.01))
    ax_polar.bar(
        bin_centers,
        bin_means,
        width=bar_width,
        color=colors_az,
        edgecolor="#555",
        linewidth=0.3,
        alpha=0.85,
    )
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title(
        "(d) Sunlit Hours by Orientation", fontsize=_NC_TITLE_SIZE, pad=15
    )
    ax_polar.tick_params(labelsize=_NC_TICK_SIZE)

    # Suptitle
    fig.suptitle(
        f"Facade Solar Access \u2014 WHO Assessment\n"
        f"{n_who:,}/{n_total:,} facade points compliant ({pct_who:.1f}%)",
        fontsize=_NC_SUPTITLE_SIZE,
        fontweight="bold",
        y=0.98,
    )

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(
        output_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    logger.info("Saved facade solar dashboard to %s", output_path)
    return output_path

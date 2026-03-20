"""Publication-quality solar access visualizations.

Provides spatial maps, distribution plots, dashboards, sun-path diagrams,
and cross-area comparison figures.  All functions follow the same save
pattern established in ``svf_v2.visualize`` and use the shared cartographic
helpers from ``src.cartography``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.cartography import (
    add_north_arrow,
    add_scale_bar,
    add_settlement_boundary,
    apply_publication_style,
    format_utm_axes,
    style_buildings_by_height,
)
from src.config import DPI, FIGURE_SIZE
from src.solar.sun import (
    compute_sun_positions,
    REFERENCE_DATES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
SOLAR_CMAP = "YlOrRd_r"
IRRADIANCE_CMAP = "inferno"

# Palette for multi-area comparisons
_AREA_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

# Palette for seasonal date tracks on sun-path diagram
_DATE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _color_for_index(idx: int) -> str:
    """Return a colour from the area palette, cycling if needed."""
    return _AREA_COLORS[idx % len(_AREA_COLORS)]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. Single-panel solar hours map
# ===================================================================


def plot_solar_access(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    title: str = "Solar Access",
    column: str = "solar_hours",
    cmap: str = SOLAR_CMAP,
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

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping solar access plot.", column)
        return Path(output_path)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Building context
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax, footprints_gdf)

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
        markersize=4,
        legend=True,
        legend_kwds={"label": "Hours of Direct Sunlight", "shrink": 0.7},
        zorder=5,
    )

    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")

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

    # Consistent colour scale
    all_vals = np.concatenate([gdf[c].dropna().values for c in date_columns])
    vmin = 0.0
    vmax = float(np.max(all_vals)) if len(all_vals) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
    axes_flat = axes.flatten()

    for idx, col in enumerate(date_columns):
        ax = axes_flat[idx]

        # Building underlay
        if footprints_gdf is not None and len(footprints_gdf) > 0:
            footprints_gdf.plot(
                ax=ax,
                facecolor="#d0d0d0",
                edgecolor="#999999",
                linewidth=0.2,
                alpha=0.5,
                zorder=0,
            )

        gdf.plot(
            ax=ax,
            column=col,
            cmap=SOLAR_CMAP,
            vmin=vmin,
            vmax=vmax,
            markersize=3,
            legend=True,
            legend_kwds={"label": "Solar hours", "shrink": 0.6},
            zorder=3,
        )

        if boundary_gdf is not None:
            add_settlement_boundary(ax, boundary_gdf)

        # Extract date label from column name
        date_label = col.replace("solar_hours_", "")
        ax.set_title(f"Solar Hours -- {date_label}", fontsize=12, fontweight="bold")
        add_scale_bar(ax)
        add_north_arrow(ax)
        ax.set_axis_off()

    # Hide unused panels
    for ax in axes_flat[len(date_columns):]:
        ax.set_visible(False)

    fig.suptitle("Seasonal Solar Access", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

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

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found -- skipping irradiance map.", column)
        return Path(output_path)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax, footprints_gdf)

    vals = gdf[column].dropna().values
    vmin = 0.0
    vmax = float(np.percentile(vals, 98)) if len(vals) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    gdf.plot(
        ax=ax,
        column=column,
        cmap=IRRADIANCE_CMAP,
        vmin=vmin,
        vmax=vmax,
        markersize=4,
        legend=True,
        legend_kwds={"label": unit, "shrink": 0.7},
        zorder=5,
    )

    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)

    ax.set_title(f"Daily Irradiance ({unit})", fontsize=13, fontweight="bold")
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
    """Distribution plot following the ``plot_svf_distribution`` pattern.

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
        bins = np.linspace(vals_nonzero.min() * 0.9, val_max * 1.05, 31)
        ax.hist(
            vals_nonzero,
            bins=bins,
            color="#f58231",
            edgecolor="white",
            alpha=0.75,
            label=f"{column} > 0",
            zorder=2,
        )

        # KDE overlay
        if len(vals_nonzero) > 5:
            try:
                kde = gaussian_kde(vals_nonzero, bw_method="scott")
                xs = np.linspace(bins[0], bins[-1], 200)
                bin_width = bins[1] - bins[0]
                ax.plot(
                    xs,
                    kde(xs) * len(vals_nonzero) * bin_width,
                    color="#e6194b",
                    linewidth=2,
                    label="KDE",
                    zorder=3,
                )
            except Exception:
                pass

    # Annotated bar for zero spike
    if n_zero > 0:
        bar_width = (val_max * 1.05 - 0) / 60 if val_max > 0 else 0.1
        ax.bar(
            0.0,
            n_zero,
            width=bar_width,
            color="#d32f2f",
            edgecolor="white",
            alpha=0.85,
            zorder=4,
        )
        ax.annotate(
            f"{pct_zero:.0f}% fully shaded\n(n={n_zero})",
            xy=(0.0, n_zero),
            xytext=(val_max * 0.15, n_zero * 0.9),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#d32f2f"),
            color="#d32f2f",
            fontweight="bold",
        )

    # Annotate fully exposed
    if n_exposed > 0:
        ax.annotate(
            f"{pct_exposed:.1f}% fully exposed\n(>{threshold_high:.1f}h, n={n_exposed})",
            xy=(0.95, 0.92),
            xycoords="axes fraction",
            fontsize=8,
            ha="right",
            color="#2e7d32",
            fontweight="bold",
        )

    # Mean/median lines
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={mean_val:.2f}")
    ax.axvline(median_val, color="orange", linestyle="-.", linewidth=1.5,
               label=f"Median={median_val:.2f}")

    # Stats box
    stats_text = (
        f"n = {n_total}\n"
        f"\u03bc = {mean_val:.3f}\n"
        f"med = {median_val:.3f}\n"
        f"\u03c3 = {std_val:.3f}\n"
        f"IQR = [{q25:.2f}, {q75:.2f}]\n"
        f"shaded = {pct_zero:.1f}%\n"
        f"exposed = {pct_exposed:.1f}%"
    )
    ax.text(
        0.98, 0.70, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(title or f"{column.replace('_', ' ').title()} Distribution")
    ax.legend(fontsize=8, loc="upper center")

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
) -> Path:
    """2x2 dashboard analogous to ``plot_svf_dashboard``.

    Panels
    ------
    * **Top-left** -- Solar hours scatter map (mean).
    * **Top-right** -- Distribution histogram + KDE.
    * **Bottom-left** -- Seasonal boxplots (one box per date if available).
    * **Bottom-right** -- Solar vs SVF scatter (if ``svf`` column present).

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
        Super-title for the figure.

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

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.4))
    ax_map, ax_hist = axes[0]
    ax_season, ax_scatter = axes[1]

    # ---- Top-left: solar hours scatter map ----
    if roads_gdf is not None and not roads_gdf.empty:
        roads_gdf.plot(ax=ax_map, color="#cccccc", linewidth=0.5, zorder=1)

    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax_map, footprints_gdf)

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
        legend_kwds={"label": "Solar hours", "shrink": 0.6},
        zorder=2,
    )
    ax_map.set_title("Solar Hours Map")
    ax_map.set_aspect("equal")
    add_scale_bar(ax_map)
    add_north_arrow(ax_map)
    format_utm_axes(ax_map, show_labels=False)

    # ---- Top-right: histogram ----
    ax_hist.hist(solar, bins=40, color="#f58231", edgecolor="white", alpha=0.8)
    ax_hist.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label="Mean")
    ax_hist.axvline(median_val, color="orange", linestyle="-.", linewidth=1.5,
                    label="Median")
    stats_text = (
        f"n = {len(solar)}\n"
        f"mean = {mean_val:.3f}\n"
        f"median = {median_val:.3f}\n"
        f"std = {std_val:.3f}\n"
        f"min = {np.min(solar):.3f}\n"
        f"max = {np.max(solar):.3f}"
    )
    ax_hist.text(
        0.98, 0.95, stats_text,
        transform=ax_hist.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    ax_hist.set_xlabel("Solar Hours")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Solar Hours Histogram")
    ax_hist.legend(fontsize=8)

    # ---- Bottom-left: seasonal boxplots ----
    date_cols = sorted([c for c in gdf.columns if c.startswith("solar_hours_20")])
    if date_cols:
        box_data = [gdf[c].dropna().values for c in date_cols]
        box_labels = [c.replace("solar_hours_", "") for c in date_cols]
        bp = ax_season.boxplot(
            box_data,
            patch_artist=True,
            showfliers=False,
        )
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_DATE_COLORS[idx % len(_DATE_COLORS)])
            patch.set_alpha(0.6)
        ax_season.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
        ax_season.set_ylabel("Solar Hours")
        ax_season.set_title("Seasonal Variation")
    else:
        ax_season.text(
            0.5, 0.5,
            "No seasonal date columns\navailable",
            transform=ax_season.transAxes,
            ha="center", va="center", fontsize=10,
        )
        ax_season.set_title("Seasonal Variation")

    # ---- Bottom-right: solar vs SVF scatter ----
    if "svf" in gdf.columns:
        svf_vals = gdf["svf"].dropna().values
        solar_vals = gdf.loc[gdf["svf"].notna(), solar_col].values
        # Align lengths
        mask = np.isfinite(svf_vals) & np.isfinite(solar_vals)
        svf_clean = svf_vals[mask]
        solar_clean = solar_vals[mask]

        ax_scatter.scatter(
            svf_clean, solar_clean,
            c="#4363d8", s=3, alpha=0.3, zorder=2,
        )

        # Linear regression
        if len(svf_clean) > 2:
            coeffs = np.polyfit(svf_clean, solar_clean, 1)
            x_line = np.linspace(0, 1, 100)
            ax_scatter.plot(x_line, np.polyval(coeffs, x_line),
                            color="red", linewidth=2, zorder=3)
            # R-squared
            ss_res = np.sum((solar_clean - np.polyval(coeffs, svf_clean)) ** 2)
            ss_tot = np.sum((solar_clean - np.mean(solar_clean)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax_scatter.annotate(
                f"R\u00b2 = {r_sq:.3f}\ny = {coeffs[0]:.2f}x + {coeffs[1]:.2f}",
                xy=(0.05, 0.92),
                xycoords="axes fraction",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax_scatter.set_xlabel("SVF")
        ax_scatter.set_ylabel("Solar Hours")
        ax_scatter.set_title("Solar Hours vs SVF")
        ax_scatter.set_xlim(0, 1)
    else:
        # CDF as fallback
        sorted_solar = np.sort(solar)
        cdf_y = np.arange(1, len(sorted_solar) + 1) / len(sorted_solar)
        ax_scatter.plot(sorted_solar, cdf_y, color="#e6194b", linewidth=1.5)
        ax_scatter.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax_scatter.axvline(median_val, color="orange", linestyle="-.",
                           linewidth=1, alpha=0.7)
        ax_scatter.set_xlabel("Solar Hours")
        ax_scatter.set_ylabel("Cumulative Probability")
        ax_scatter.set_title("Solar Hours CDF")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

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

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Polar setup: theta=0 at North, clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    )
    ax.set_rlim(0, 90)
    ax.set_rlabel_position(135)
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(
        ["90\u00b0", "75\u00b0", "60\u00b0", "45\u00b0", "30\u00b0", "15\u00b0", "0\u00b0"],
        fontsize=7,
    )

    for idx, date in enumerate(dates):
        color = _DATE_COLORS[idx % len(_DATE_COLORS)]
        positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=5,
            hour_end=19,
            interval_minutes=10,  # fine resolution for smooth track
        )
        if not positions:
            continue

        alts = [p[0] for p in positions]
        azs = [p[1] for p in positions]
        theta = np.radians(azs)
        r = 90.0 - np.array(alts)  # zenith angle

        ax.plot(theta, r, color=color, linewidth=2, label=date, zorder=3)

        # Hour markers (every 60 min = every 6th point at 10-min resolution)
        hour_positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=5,
            hour_end=19,
            interval_minutes=60,
        )
        for alt_h, az_h in hour_positions:
            th = np.radians(az_h)
            rh = 90.0 - alt_h
            ax.plot(th, rh, "o", color=color, markersize=5, zorder=4)
            # Approximate solar hour from azimuth/altitude context
            # Label the hour based on position index
        # Label hours at start/middle/end
        if len(hour_positions) >= 1:
            th0 = np.radians(hour_positions[0][1])
            r0 = 90.0 - hour_positions[0][0]
            ax.annotate("6h", xy=(th0, r0), fontsize=6, color=color,
                         fontweight="bold", ha="center")
        if len(hour_positions) >= 7:
            th12 = np.radians(hour_positions[6][1])
            r12 = 90.0 - hour_positions[6][0]
            ax.annotate("12h", xy=(th12, r12), fontsize=6, color=color,
                         fontweight="bold", ha="center")
        if len(hour_positions) >= 13:
            th18 = np.radians(hour_positions[12][1])
            r18 = 90.0 - hour_positions[12][0]
            ax.annotate("18h", xy=(th18, r18), fontsize=6, color=color,
                         fontweight="bold", ha="center")

    ax.set_title(
        f"Sun Path Diagram\n(lat={latitude:.2f}\u00b0, lon={longitude:.2f}\u00b0)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=9)

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
    """Scatter plot of SVF vs solar hours with linear regression line.

    Colours points by local density and annotates R-squared and the
    regression equation.

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
            solar_col, svf_col,
        )
        return Path(output_path)

    data = gdf[[svf_col, solar_col]].dropna()
    svf = data[svf_col].values
    solar = data[solar_col].values

    if len(svf) < 3:
        logger.warning("Too few points for solar vs SVF scatter -- skipping.")
        return Path(output_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Density-based colouring via hexbin
    hb = ax.hexbin(
        svf, solar,
        gridsize=40,
        cmap="YlOrRd",
        mincnt=1,
        zorder=2,
    )
    plt.colorbar(hb, ax=ax, label="Point count")

    # Linear regression
    coeffs = np.polyfit(svf, solar, 1)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color="blue", linewidth=2, linestyle="--", zorder=3, label="Linear fit")

    # R-squared
    ss_res = np.sum((solar - np.polyval(coeffs, svf)) ** 2)
    ss_tot = np.sum((solar - np.mean(solar)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    ax.annotate(
        f"R\u00b2 = {r_sq:.3f}\ny = {coeffs[0]:.2f}x + {coeffs[1]:.2f}\nn = {len(svf)}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("SVF", fontsize=11)
    ax.set_ylabel("Solar Hours (mean)", fontsize=11)
    ax.set_title("Solar Access vs Sky View Factor", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

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
    """Cross-area comparison following ``plot_svf_comparison``.

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
    fig, (ax_kde, ax_box) = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]),
    )

    area_names = list(areas_data.keys())
    colors = [_color_for_index(i) for i in range(len(area_names))]
    arrays: list[np.ndarray] = []

    # ---- Left panel: KDE curves ----
    for idx, (name, gdf) in enumerate(areas_data.items()):
        if column not in gdf.columns:
            logger.warning("Column '%s' missing in area '%s' -- skipping.", column, name)
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
        ax_kde.fill_between(xs, kde(xs), alpha=0.15, color=colors[idx])
        ax_kde.axvline(mean_val, color=colors[idx], linestyle="--",
                       linewidth=1, alpha=0.7)

    ax_kde.set_xlabel(column.replace("_", " ").title())
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"{column.replace('_', ' ').title()} (KDE)")

    # Legend with stats
    legend_handles = []
    for idx, name in enumerate(area_names):
        vals = arrays[idx]
        if len(vals) == 0:
            continue
        label = (
            f"{name}  "
            f"(mean={np.mean(vals):.2f}, "
            f"med={np.median(vals):.2f}, "
            f"n={len(vals)})"
        )
        legend_handles.append(
            Line2D([0], [0], color=colors[idx], linewidth=2, label=label)
        )
    ax_kde.legend(handles=legend_handles, fontsize=8, loc="upper left")

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
            body.set_alpha(0.3)

        bp = ax_box.boxplot(
            valid_arrays,
            positions=range(len(valid_names)),
            widths=0.2,
            patch_artist=True,
            showfliers=False,
        )
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(valid_colors[idx])
            patch.set_alpha(0.6)

        ax_box.set_xticks(range(len(valid_names)))
        ax_box.set_xticklabels(valid_names, rotation=30, ha="right", fontsize=9)

        # Annotate mean/median/n
        for idx, name in enumerate(valid_names):
            vals = valid_arrays[idx]
            ax_box.annotate(
                f"mean={np.mean(vals):.2f}\nmed={np.median(vals):.2f}\nn={len(vals)}",
                xy=(idx, np.mean(vals)),
                xytext=(idx + 0.35, np.mean(vals)),
                fontsize=7, ha="left", va="center",
                color=valid_colors[idx],
            )

    ax_box.set_ylabel(column.replace("_", " ").title())
    ax_box.set_title(f"{column.replace('_', ' ').title()} (Box + Violin)")

    fig.suptitle("Comparative Solar Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved solar comparison to %s", output_path)
    return output_path

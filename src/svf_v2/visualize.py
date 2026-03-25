"""Visualization functions for SVF v2 results.

Core plot functions (heatmap, street map, facade orientation/height) are
used by ``io.py`` during result export.  Comparative and dashboard functions
are intended for post-hoc analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.cartography import (
    add_north_arrow,
    add_scale_bar,
    add_settlement_boundary,
    add_terrain_contours,
    format_utm_axes,
    get_svf_cmap,
    style_buildings_by_height,
)
from src.config import DPI, FIGURE_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive line/marker sizing based on study area extent
# ---------------------------------------------------------------------------


def _adaptive_linewidth(gdf: gpd.GeoDataFrame, lw_min: float = 0.5, lw_max: float = 3.0) -> float:
    """Compute a linewidth that scales inversely with the study area extent.

    Small areas (diagonal ~0.5 km) get thick lines (~3.0), large areas
    (diagonal ~5 km) get thin lines (~0.8).  The formula is::

        lw = clamp(3.0 / (diag_km + 0.5), lw_min, lw_max)

    Parameters
    ----------
    gdf : GeoDataFrame
        Any GeoDataFrame in a projected CRS (units = metres).
    lw_min, lw_max : float
        Clamping bounds for the returned linewidth.

    Returns
    -------
    float
        Adaptive linewidth value.
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    diag_km = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2) / 1000.0
    lw = 3.0 / (diag_km + 0.5)
    return float(np.clip(lw, lw_min, lw_max))


def _adaptive_markersize(gdf: gpd.GeoDataFrame, ms_min: float = 0.5, ms_max: float = 5.0) -> float:
    """Compute a marker size that scales inversely with the study area extent.

    Same logic as :func:`_adaptive_linewidth` but with a different default
    range appropriate for scatter marker sizes.
    """
    bounds = gdf.total_bounds
    diag_km = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2) / 1000.0
    ms = 4.0 / (diag_km + 0.3)
    return float(np.clip(ms, ms_min, ms_max))


def _adaptive_folium_weight(gdf: gpd.GeoDataFrame) -> float:
    """Compute a Folium polyline weight that scales inversely with extent.

    Returns a value in [1.5, 5.0].
    """
    bounds = gdf.total_bounds
    diag_km = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2) / 1000.0
    w = 4.0 / (diag_km + 0.5)
    return float(np.clip(w, 1.5, 5.0))


# ---------------------------------------------------------------------------
# Core plot functions (called by io.py save_*_results)
# ---------------------------------------------------------------------------


def plot_svf_heatmap(
    points: np.ndarray,
    svf: np.ndarray,
    output_path: Path,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Scatter-plot heatmap of grid SVF values with optional building footprints."""
    fig, ax = plt.subplots(figsize=(12, 10))

    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax, footprints_gdf)

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=svf,
        cmap=get_svf_cmap(),
        s=2,
        vmin=0,
        vmax=1,
        alpha=0.8,
        zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="SVF")
    ax.set_aspect("equal")
    ax.set_title("Sky View Factor (Grid)")
    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


def plot_street_svf(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
    segments_gdf: Optional[gpd.GeoDataFrame] = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    dtm_path=None,
    mode: str = "auto",
):
    """Street SVF map with building footprints and cartographic elements.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point-level SVF data with ``svf`` column.
    output_path : Path
        Where to save the figure.
    roads_gdf, footprints_gdf : GeoDataFrame, optional
        Context layers.
    segments_gdf : GeoDataFrame, optional
        Line geometries with ``svf_mean`` column for segment rendering.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay.
    dtm_path : str or Path, optional
        DTM raster for terrain contours.
    mode : str
        'points' — always scatter; 'segments' — always line;
        'auto' — segments if available, else points.
    """
    cmap = get_svf_cmap()
    use_segments = (
        segments_gdf is not None
        and len(segments_gdf) > 0
        and "svf_mean" in segments_gdf.columns
        and mode != "points"
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    # Terrain contours (lowest layer)
    if dtm_path is not None:
        add_terrain_contours(ax, dtm_path)

    # Building footprints
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax, footprints_gdf)

    # Road centrelines (subtle)
    if roads_gdf is not None and not use_segments:
        roads_gdf.plot(ax=ax, color="#aaaaaa", linewidth=0.5, zorder=2)

    # Main SVF layer — adaptive sizing based on study area extent
    if use_segments:
        lw = _adaptive_linewidth(segments_gdf)
        logger.debug("  Adaptive linewidth=%.2f for segments", lw)
        segments_gdf.plot(
            ax=ax,
            column="svf_mean",
            cmap=cmap,
            linewidth=lw,
            vmin=0,
            vmax=1,
            legend=True,
            legend_kwds={"label": "SVF (segment mean)", "shrink": 0.6},
            zorder=3,
        )
        ax.set_title("Street SVF — Segment Mean")
    else:
        ms = _adaptive_markersize(gdf)
        logger.debug("  Adaptive markersize=%.2f for points", ms)
        gdf.plot(
            ax=ax,
            column="svf",
            cmap=cmap,
            markersize=ms,
            vmin=0,
            vmax=1,
            legend=True,
            legend_kwds={"label": "SVF", "shrink": 0.6},
            zorder=3,
        )
        ax.set_title("Street SVF")

    # Settlement boundary
    if boundary_gdf is not None:
        add_settlement_boundary(ax, boundary_gdf)

    ax.set_aspect("equal")
    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


def plot_facade_by_orientation(gdf: gpd.GeoDataFrame, output_path: Path):
    """Polar scatter of facade SVF coloured by azimuth."""
    if "svf" not in gdf.columns or "facade_azimuth" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "polar"})
    az_rad = np.radians(gdf["facade_azimuth"].values)
    sc = ax.scatter(
        az_rad,
        gdf["svf"].values,
        c=gdf["svf"].values,
        cmap=get_svf_cmap(),
        s=1,
        alpha=0.3,
        vmin=0,
        vmax=1,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                      labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title("Facade SVF by Orientation")
    plt.colorbar(sc, ax=ax, label="SVF", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


def plot_facade_by_height(gdf: gpd.GeoDataFrame, output_path: Path):
    """Scatter of facade SVF vs height above ground with LOWESS trend."""
    if "svf" not in gdf.columns or "height_above_ground" not in gdf.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        gdf["height_above_ground"].values,
        gdf["svf"].values,
        c=gdf["facade_azimuth"].values
        if "facade_azimuth" in gdf.columns
        else "steelblue",
        cmap="hsv",
        s=1,
        alpha=0.3,
    )
    ax.set_xlabel("Height above ground (m)")
    ax.set_ylabel("SVF")
    ax.set_title("Facade SVF vs Height")
    ax.set_ylim(0, 1)
    if "facade_azimuth" in gdf.columns:
        plt.colorbar(sc, ax=ax, label="Azimuth")

    # LOWESS trend line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        h = gdf["height_above_ground"].values
        s = gdf["svf"].values
        mask = np.isfinite(h) & np.isfinite(s)
        if mask.sum() > 20:
            result = lowess(s[mask], h[mask], frac=0.3, return_sorted=True)
            ax.plot(result[:, 0], result[:, 1], color="red", linewidth=2,
                    label="LOWESS trend", zorder=5)
            ax.legend(fontsize=8)
    except ImportError:
        pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


# ---------------------------------------------------------------------------
# Distribution plot (bimodal-aware)
# ---------------------------------------------------------------------------


def plot_svf_distribution(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    svf_col: str = "svf",
    title: str | None = None,
) -> Path:
    """Bimodal-aware SVF distribution plot.

    Handles the common spike at SVF=0 by separating it as an annotated bar,
    then showing a histogram + KDE for the non-zero portion.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *svf_col*.
    output_path : Path
        Where to save the figure.
    svf_col : str
        Column with SVF values.
    title : str, optional
        Figure title. Defaults to 'SVF Distribution'.

    Returns
    -------
    Path
        *output_path* on success.
    """
    svf = gdf[svf_col].dropna().values
    n_total = len(svf)
    if n_total == 0:
        logger.warning("No SVF data for distribution plot — skipping.")
        return output_path

    # Separate zero and non-zero
    n_zero = int(np.sum(svf == 0.0))
    pct_zero = 100.0 * n_zero / n_total
    svf_nonzero = svf[svf > 0.0]
    n_open = int(np.sum(svf > 0.9))
    pct_open = 100.0 * n_open / n_total

    mean_val = float(np.mean(svf))
    median_val = float(np.median(svf))
    std_val = float(np.std(svf))
    q25, q75 = float(np.percentile(svf, 25)), float(np.percentile(svf, 75))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main histogram for non-zero values
    if len(svf_nonzero) > 0:
        bins = np.linspace(0.01, 1.0, 31)
        ax.hist(
            svf_nonzero,
            bins=bins,
            color="#4363d8",
            edgecolor="white",
            alpha=0.75,
            label="SVF > 0",
            zorder=2,
        )

        # KDE overlay
        if len(svf_nonzero) > 5:
            try:
                kde = gaussian_kde(svf_nonzero, bw_method="scott")
                xs = np.linspace(0.01, 1.0, 200)
                # Scale KDE to histogram height
                bin_width = bins[1] - bins[0]
                ax.plot(
                    xs,
                    kde(xs) * len(svf_nonzero) * bin_width,
                    color="#e6194b",
                    linewidth=2,
                    label="KDE",
                    zorder=3,
                )
            except Exception:
                pass

    # Annotated bar for SVF=0 spike
    if n_zero > 0:
        bar_width = 0.01
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
            f"{pct_zero:.0f}% occluded\n(SVF=0, n={n_zero})",
            xy=(0.0, n_zero),
            xytext=(0.08, n_zero * 0.9),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#d32f2f"),
            color="#d32f2f",
            fontweight="bold",
        )

    # Annotate open-sky fraction
    if n_open > 0:
        ax.annotate(
            f"{pct_open:.1f}% open sky\n(SVF>0.9, n={n_open})",
            xy=(0.95, 0.92),
            xycoords="axes fraction",
            fontsize=8,
            ha="right",
            color="#2e7d32",
            fontweight="bold",
        )

    # Mean/median lines
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.2f}")
    ax.axvline(
        median_val, color="orange", linestyle="-.", linewidth=1.5, label=f"Median={median_val:.2f}"
    )

    # Stats box
    stats_text = (
        f"n = {n_total}\n"
        f"\u03bc = {mean_val:.3f}\n"
        f"med = {median_val:.3f}\n"
        f"\u03c3 = {std_val:.3f}\n"
        f"IQR = [{q25:.2f}, {q75:.2f}]\n"
        f"occluded = {pct_zero:.1f}%\n"
        f"open = {pct_open:.1f}%"
    )
    ax.text(
        0.98,
        0.70,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("SVF")
    ax.set_ylabel("Count")
    ax.set_title(title or "SVF Distribution")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(-0.03, 1.03)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Interactive Folium map
# ---------------------------------------------------------------------------


def plot_svf_interactive(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    segments_gdf: Optional[gpd.GeoDataFrame] = None,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
    svf_col: str = "svf",
) -> Path:
    """Create an interactive Folium map of SVF results.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point-level SVF data.
    output_path : Path
        Where to save the HTML file.
    segments_gdf : GeoDataFrame, optional
        Segment lines with ``svf_mean`` for coloured line rendering.
    footprints_gdf : GeoDataFrame, optional
        Building footprints as semi-transparent overlay.
    svf_col : str
        SVF column name in *gdf*.

    Returns
    -------
    Path
        *output_path* on success.
    """
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError:
        logger.warning("folium not installed — skipping interactive map")
        return output_path

    # Reproject to 4326
    gdf_4326 = gdf.to_crs(epsg=4326)
    center = [gdf_4326.geometry.y.mean(), gdf_4326.geometry.x.mean()]

    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")

    # Try adding satellite tiles
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
    ).add_to(m)

    # Build a cividis colormap from matplotlib
    cividis_mpl = plt.colormaps["cividis"]
    cividis_colors = [
        "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in cividis_mpl(v)[:3]])
        for v in np.linspace(0, 1, 10)
    ]
    colormap = LinearColormap(cividis_colors, vmin=0, vmax=1, caption="SVF")
    colormap.add_to(m)

    # Building footprints (toggleable)
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        fp_4326 = footprints_gdf.to_crs(epsg=4326)
        buildings_layer = folium.FeatureGroup(name="Buildings", show=True)
        folium.GeoJson(
            fp_4326,
            style_function=lambda _: {
                "fillColor": "#888888",
                "color": "#555555",
                "weight": 0.5,
                "fillOpacity": 0.3,
            },
        ).add_to(buildings_layer)
        buildings_layer.add_to(m)

    # Segment lines coloured by svf_mean — adaptive weight
    if segments_gdf is not None and len(segments_gdf) > 0 and "svf_mean" in segments_gdf.columns:
        seg_4326 = segments_gdf.to_crs(epsg=4326)
        folium_weight = _adaptive_folium_weight(segments_gdf)
        logger.debug("  Adaptive Folium weight=%.2f", folium_weight)
        seg_layer = folium.FeatureGroup(name="SVF Segments", show=True)
        for _, row in seg_4326.iterrows():
            svf_val = float(row["svf_mean"])
            color = "#{:02x}{:02x}{:02x}".format(
                *[int(c * 255) for c in plt.colormaps["cividis"](svf_val)[:3]]
            )
            popup_text = (
                f"SVF mean: {svf_val:.3f}<br>"
                f"SVF median: {row.get('svf_median', 'N/A')}<br>"
                f"Points: {row.get('n_points', 'N/A')}"
            )
            geom = row.geometry
            if geom.geom_type == "MultiLineString":
                line_coords = [
                    [(c[1], c[0]) for c in part.coords]
                    for part in geom.geoms
                ]
            else:
                line_coords = [[(c[1], c[0]) for c in geom.coords]]
            for coords in line_coords:
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=folium_weight,
                    opacity=0.8,
                    popup=folium.Popup(popup_text, max_width=200),
                ).add_to(seg_layer)
        seg_layer.add_to(m)

    folium.LayerControl().add_to(m)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("  Saved %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Colour palette for multi-area comparisons
# ---------------------------------------------------------------------------
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


def _color_for_index(idx: int) -> str:
    """Return a colour from the palette, cycling if needed."""
    return _AREA_COLORS[idx % len(_AREA_COLORS)]


# ===================================================================
# 1. Comparative SVF distribution across areas
# ===================================================================


def plot_svf_comparison(
    areas_data: dict[str, gpd.GeoDataFrame],
    output_path: Path,
) -> Path:
    """Create a comparative SVF distribution figure for multiple areas.

    Produces a 1x2 figure:
      - Left panel: overlaid KDE curves for each area.
      - Right panel: box-and-violin plots per area.
    Both panels are annotated with mean, median, and sample size.

    Parameters
    ----------
    areas_data : dict[str, gpd.GeoDataFrame]
        Mapping of area name to a GeoDataFrame that contains an ``svf``
        column with numeric Sky View Factor values.
    output_path : Path
        Destination file path for the saved figure (e.g. ``*.png``).

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    fig, (ax_kde, ax_box) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]),
    )

    area_names = list(areas_data.keys())
    colors = [_color_for_index(i) for i in range(len(area_names))]
    svf_arrays: list[np.ndarray] = []

    # ---- Left panel: KDE curves ----
    for idx, (name, gdf) in enumerate(areas_data.items()):
        svf = gdf["svf"].dropna().values
        svf_arrays.append(svf)

        mean_val = float(np.mean(svf))

        kde = gaussian_kde(svf, bw_method="scott")
        xs = np.linspace(0, 1, 300)
        ax_kde.plot(xs, kde(xs), color=colors[idx], linewidth=2, label=name)
        ax_kde.fill_between(xs, kde(xs), alpha=0.15, color=colors[idx])

        # Dashed vertical line at the mean
        ax_kde.axvline(
            mean_val,
            color=colors[idx],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

    ax_kde.set_xlabel("SVF")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title("SVF Distribution (KDE)")
    ax_kde.set_xlim(0, 1)

    # Build legend with stats
    legend_handles = []
    for idx, name in enumerate(area_names):
        svf = svf_arrays[idx]
        label = (
            f"{name}  "
            f"(mean={np.mean(svf):.2f}, "
            f"med={np.median(svf):.2f}, "
            f"n={len(svf)})"
        )
        legend_handles.append(
            Line2D([0], [0], color=colors[idx], linewidth=2, label=label)
        )
    ax_kde.legend(handles=legend_handles, fontsize=8, loc="upper left")

    # ---- Right panel: violin + box ----
    parts = ax_box.violinplot(
        svf_arrays,
        positions=range(len(area_names)),
        showextrema=False,
        showmedians=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_alpha(0.3)

    bp = ax_box.boxplot(
        svf_arrays,
        positions=range(len(area_names)),
        widths=0.2,
        patch_artist=True,
        showfliers=False,
    )
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.6)

    ax_box.set_xticks(range(len(area_names)))
    ax_box.set_xticklabels(area_names, rotation=30, ha="right", fontsize=9)
    ax_box.set_ylabel("SVF")
    ax_box.set_ylim(0, 1)
    ax_box.set_title("SVF Distribution (Box + Violin)")

    # Annotate mean/median/n on the violin panel
    for idx, name in enumerate(area_names):
        svf = svf_arrays[idx]
        ax_box.annotate(
            f"mean={np.mean(svf):.2f}\nmed={np.median(svf):.2f}\nn={len(svf)}",
            xy=(idx, np.mean(svf)),
            xytext=(idx + 0.35, np.mean(svf)),
            fontsize=7,
            ha="left",
            va="center",
            color=colors[idx],
        )

    fig.suptitle("Comparative SVF Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ===================================================================
# 2. Single-area SVF dashboard (2x2)
# ===================================================================


def plot_svf_dashboard(
    street_gdf: gpd.GeoDataFrame,
    output_path: Path,
    roads_gdf: gpd.GeoDataFrame | None = None,
    title: str | None = None,
) -> Path:
    """Generate a 2x2 SVF dashboard for a single study area.

    Panels
    ------
    * **Top-left** -- Street SVF map (scatter coloured by SVF, 0--1 range,
      ``RdYlGn`` colourmap). If *roads_gdf* is provided, road geometries
      are drawn as a background layer.
    * **Top-right** -- SVF histogram with vertical mean/median lines and a
      text annotation box summarising descriptive statistics.
    * **Bottom-left** -- Per-segment SVF box-plot (top 20 segments by
      point count). Requires a ``street_id`` column; if absent the panel
      shows a message instead.
    * **Bottom-right** -- Empirical cumulative distribution function (CDF)
      of SVF.

    Parameters
    ----------
    street_gdf : gpd.GeoDataFrame
        GeoDataFrame with point geometries and an ``svf`` column.
    output_path : Path
        File path to save the figure.
    roads_gdf : gpd.GeoDataFrame | None
        Optional road centreline geometries drawn behind the scatter.
    title : str | None
        Optional super-title for the figure.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    svf = street_gdf["svf"].dropna().values
    mean_val = float(np.mean(svf))
    median_val = float(np.median(svf))
    std_val = float(np.std(svf))

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.4))
    ax_map, ax_hist = axes[0]
    ax_seg, ax_cdf = axes[1]

    # ---- Top-left: SVF scatter map (adaptive marker size) ----
    ms = _adaptive_markersize(street_gdf)
    logger.debug("  Dashboard adaptive markersize=%.2f", ms)

    if roads_gdf is not None and not roads_gdf.empty:
        roads_gdf.plot(ax=ax_map, color="#cccccc", linewidth=0.5, zorder=1)

    street_gdf.plot(
        ax=ax_map,
        column="svf",
        cmap=get_svf_cmap(),
        vmin=0,
        vmax=1,
        markersize=ms,
        legend=True,
        legend_kwds={"label": "SVF", "shrink": 0.6},
        zorder=2,
    )
    ax_map.set_title("Street SVF Map")
    ax_map.set_aspect("equal")
    add_scale_bar(ax_map)
    add_north_arrow(ax_map)
    format_utm_axes(ax_map, show_labels=False)

    # ---- Top-right: histogram ----
    ax_hist.hist(svf, bins=40, color="#4363d8", edgecolor="white", alpha=0.8)
    ax_hist.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label="Mean")
    ax_hist.axvline(
        median_val, color="orange", linestyle="-.", linewidth=1.5, label="Median"
    )
    stats_text = (
        f"n = {len(svf)}\n"
        f"mean = {mean_val:.3f}\n"
        f"median = {median_val:.3f}\n"
        f"std = {std_val:.3f}\n"
        f"min = {np.min(svf):.3f}\n"
        f"max = {np.max(svf):.3f}"
    )
    ax_hist.text(
        0.98,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    ax_hist.set_xlabel("SVF")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("SVF Histogram")
    ax_hist.legend(fontsize=8)

    # ---- Bottom-left: per-segment boxplot ----
    if "street_id" in street_gdf.columns:
        counts = street_gdf.groupby("street_id").size()
        top_segments = counts.nlargest(20).index
        seg_data = [
            street_gdf.loc[street_gdf["street_id"] == sid, "svf"].dropna().values
            for sid in top_segments
        ]
        seg_labels = [str(s) for s in top_segments]

        if seg_data:
            ax_seg.boxplot(seg_data, patch_artist=True, showfliers=False)
            ax_seg.set_xticklabels(seg_labels, rotation=60, ha="right", fontsize=6)
            ax_seg.set_ylabel("SVF")
            ax_seg.set_ylim(0, 1)
            ax_seg.set_title("SVF by Street Segment (Top 20)")
        else:
            ax_seg.text(
                0.5,
                0.5,
                "No segment data",
                transform=ax_seg.transAxes,
                ha="center",
                va="center",
            )
            ax_seg.set_title("SVF by Street Segment")
    else:
        ax_seg.text(
            0.5,
            0.5,
            "No 'street_id' column\nin GeoDataFrame",
            transform=ax_seg.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax_seg.set_title("SVF by Street Segment")

    # ---- Bottom-right: CDF ----
    sorted_svf = np.sort(svf)
    cdf_y = np.arange(1, len(sorted_svf) + 1) / len(sorted_svf)
    ax_cdf.plot(sorted_svf, cdf_y, color="#e6194b", linewidth=1.5)
    ax_cdf.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_cdf.axvline(median_val, color="orange", linestyle="-.", linewidth=1, alpha=0.7)
    ax_cdf.set_xlabel("SVF")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("SVF Cumulative Distribution (CDF)")
    ax_cdf.set_xlim(0, 1)
    ax_cdf.set_ylim(0, 1)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path

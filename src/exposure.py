"""Zone-level environmental exposure metrics and composite index.

Computes solar deficit, SVF deficit, building density, and a configurable
composite exposure index at the analysis-zone scale (grid cells from
``urban_morphology.py``).  All aggregations use spatial joins or overlays
-- never pixel-level point-sampling loops.

The exposure index is a transparent, weighted composite that highlights
zones of compounded environmental deprivation.  No causality is inferred;
the index is a spatial screening tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.cartography import add_north_arrow, add_scale_bar, add_settlement_boundary
from src.config import DPI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Solar deficit
# ---------------------------------------------------------------------------


def compute_zone_solar_deficit(
    zones_gdf: gpd.GeoDataFrame,
    solar_gdf: gpd.GeoDataFrame,
    solar_col: str = "solar_hours",
    reference_hours: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Aggregate solar access points to zones and compute solar deficit.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Analysis zones with ``zone_id`` and ``geometry``.
    solar_gdf : GeoDataFrame
        Point features with a solar-hours column.
    solar_col : str
        Column in *solar_gdf* holding solar-hours values.
    reference_hours : float, optional
        Maximum expected solar hours.  If *None*, the 90th percentile of
        zone-level means is used as reference.

    Returns
    -------
    GeoDataFrame
        Copy of *zones_gdf* with added columns: ``mean_solar``,
        ``min_solar``, ``solar_deficit``.
    """
    zones = zones_gdf.copy()

    if solar_gdf is None or solar_gdf.empty or solar_col not in solar_gdf.columns:
        zones["mean_solar"] = np.nan
        zones["min_solar"] = np.nan
        zones["solar_deficit"] = np.nan
        logger.warning("No solar data available -- solar columns set to NaN.")
        return zones

    # Ensure matching CRS
    if (
        solar_gdf.crs is not None
        and zones.crs is not None
        and solar_gdf.crs != zones.crs
    ):
        solar_gdf = solar_gdf.to_crs(zones.crs)

    joined = gpd.sjoin(
        solar_gdf[[solar_col, "geometry"]],
        zones[["zone_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        zones["mean_solar"] = np.nan
        zones["min_solar"] = np.nan
        zones["solar_deficit"] = np.nan
        logger.warning("Spatial join returned no matches -- solar columns set to NaN.")
        return zones

    agg = joined.groupby("zone_id")[solar_col].agg(mean_solar="mean", min_solar="min")

    zones = zones.merge(agg, on="zone_id", how="left")

    # Determine reference
    if reference_hours is None:
        valid_means = zones["mean_solar"].dropna()
        if len(valid_means) > 0:
            reference_hours = float(np.percentile(valid_means, 90))
        else:
            reference_hours = 1.0  # fallback to avoid division by zero

    if reference_hours <= 0:
        reference_hours = 1.0

    zones["solar_deficit"] = (1.0 - zones["mean_solar"] / reference_hours).clip(
        0.0, 1.0
    )

    logger.info(
        "Solar deficit computed for %d zones (reference=%.1f h).",
        zones["solar_deficit"].notna().sum(),
        reference_hours,
    )
    return zones


# ---------------------------------------------------------------------------
# 2. SVF deficit
# ---------------------------------------------------------------------------


def compute_zone_svf_deficit(
    zones_gdf: gpd.GeoDataFrame,
    svf_gdf: gpd.GeoDataFrame,
    svf_col: str = "svf",
) -> gpd.GeoDataFrame:
    """Aggregate SVF points to zones and compute SVF deficit.

    SVF is already normalised to [0, 1], so the deficit is simply
    ``1 - mean_svf``.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Analysis zones with ``zone_id`` and ``geometry``.
    svf_gdf : GeoDataFrame
        Point features with an SVF column.
    svf_col : str
        Column in *svf_gdf* holding SVF values.

    Returns
    -------
    GeoDataFrame
        Copy of *zones_gdf* with added columns: ``mean_svf``,
        ``median_svf``, ``svf_deficit``.
    """
    zones = zones_gdf.copy()

    if svf_gdf is None or svf_gdf.empty or svf_col not in svf_gdf.columns:
        zones["mean_svf"] = np.nan
        zones["median_svf"] = np.nan
        zones["svf_deficit"] = np.nan
        logger.warning("No SVF data available -- SVF columns set to NaN.")
        return zones

    if svf_gdf.crs is not None and zones.crs is not None and svf_gdf.crs != zones.crs:
        svf_gdf = svf_gdf.to_crs(zones.crs)

    joined = gpd.sjoin(
        svf_gdf[[svf_col, "geometry"]],
        zones[["zone_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        zones["mean_svf"] = np.nan
        zones["median_svf"] = np.nan
        zones["svf_deficit"] = np.nan
        logger.warning("Spatial join returned no matches -- SVF columns set to NaN.")
        return zones

    agg = joined.groupby("zone_id")[svf_col].agg(mean_svf="mean", median_svf="median")

    zones = zones.merge(agg, on="zone_id", how="left")

    zones["svf_deficit"] = (1.0 - zones["mean_svf"]).clip(0.0, 1.0)

    logger.info(
        "SVF deficit computed for %d zones.", zones["svf_deficit"].notna().sum()
    )
    return zones


# ---------------------------------------------------------------------------
# 3. Building density (volumetric)
# ---------------------------------------------------------------------------


def compute_zone_building_density(
    zones_gdf: gpd.GeoDataFrame,
    footprints_gdf: gpd.GeoDataFrame,
    height_col: str = "altura",
) -> gpd.GeoDataFrame:
    """Compute volumetric building density per zone via spatial overlay.

    Uses ``gpd.overlay(how="intersection")`` to clip footprints to zone
    boundaries, avoiding any point-sampling loops.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Analysis zones with ``zone_id``, ``zone_area``, and ``geometry``.
    footprints_gdf : GeoDataFrame
        Building footprints with a height column.
    height_col : str
        Column in *footprints_gdf* holding building heights.

    Returns
    -------
    GeoDataFrame
        Copy of *zones_gdf* with added columns: ``built_volume``,
        ``open_space_ratio``, ``density_ratio``.
    """
    zones = zones_gdf.copy()

    if footprints_gdf is None or footprints_gdf.empty:
        zones["built_volume"] = 0.0
        zones["open_space_ratio"] = 1.0
        zones["density_ratio"] = 0.0
        logger.warning("No footprints provided -- density columns set to zero.")
        return zones

    if (
        footprints_gdf.crs is not None
        and zones.crs is not None
        and footprints_gdf.crs != zones.crs
    ):
        footprints_gdf = footprints_gdf.to_crs(zones.crs)

    # Prepare footprints: keep only geometry and height
    fp = footprints_gdf.copy()
    if height_col not in fp.columns:
        fp[height_col] = 1.0  # default to 1 m if no height data
        logger.warning(
            "Height column '%s' not found -- defaulting to 1.0 m.", height_col
        )

    # Replace invalid heights
    fp[height_col] = fp[height_col].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    fp[height_col] = fp[height_col].clip(lower=0.0)

    # Assign a temporary building id for the overlay
    fp = fp[["geometry", height_col]].copy()
    fp["_bld_id"] = range(len(fp))

    # Ensure zone_area exists
    if "zone_area" not in zones.columns:
        zones["zone_area"] = zones.geometry.area

    # Spatial overlay: clip footprints to zone boundaries
    overlay = gpd.overlay(
        zones[["zone_id", "zone_area", "geometry"]],
        fp,
        how="intersection",
        keep_geom_type=False,
    )

    if overlay.empty:
        zones["built_volume"] = 0.0
        zones["open_space_ratio"] = 1.0
        zones["density_ratio"] = 0.0
        logger.warning(
            "Overlay returned no intersections -- density columns set to zero."
        )
        return zones

    # Compute clipped footprint area and volume
    overlay["_clipped_area"] = overlay.geometry.area
    overlay["_clipped_volume"] = overlay["_clipped_area"] * overlay[height_col]

    # Aggregate per zone
    agg = overlay.groupby("zone_id").agg(
        built_volume=("_clipped_volume", "sum"),
        _built_area=("_clipped_area", "sum"),
    )

    zones = zones.merge(agg, on="zone_id", how="left")
    zones["built_volume"] = zones["built_volume"].fillna(0.0)
    zones["_built_area"] = zones["_built_area"].fillna(0.0)

    # Open space ratio: proportion of zone not covered by buildings
    zones["open_space_ratio"] = np.where(
        zones["zone_area"] > 0,
        1.0 - zones["_built_area"] / zones["zone_area"],
        1.0,
    )
    zones["open_space_ratio"] = zones["open_space_ratio"].clip(0.0, 1.0)

    # Density ratio: volumetric intensity = built_volume / zone_area
    zones["density_ratio"] = np.where(
        zones["zone_area"] > 0,
        zones["built_volume"] / zones["zone_area"],
        0.0,
    )

    zones = zones.drop(columns=["_built_area"], errors="ignore")

    logger.info(
        "Building density computed for %d zones (mean density_ratio=%.2f).",
        len(zones),
        zones["density_ratio"].mean(),
    )
    return zones


# ---------------------------------------------------------------------------
# 4. Composite exposure index
# ---------------------------------------------------------------------------


def compute_exposure_index(
    zones_gdf: gpd.GeoDataFrame,
    metrics: dict[str, str],
    weights: Optional[dict[str, float]] = None,
) -> gpd.GeoDataFrame:
    """Weighted composite exposure index from normalised metrics.

    Each metric column is min-max normalised to [0, 1], then combined via
    a weighted sum.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Must contain the columns listed in *metrics* values.
    metrics : dict
        Mapping of human-readable names to column names, e.g.
        ``{"solar": "solar_deficit", "svf": "svf_deficit", "density": "density_ratio"}``.
    weights : dict, optional
        Mapping of metric names to weights.  Weights are normalised
        internally so they sum to 1.  If *None*, equal weights are used.

    Returns
    -------
    GeoDataFrame
        Copy of *zones_gdf* with an added ``exposure_index`` column
        (0 = low exposure, 1 = high exposure).
    """
    zones = zones_gdf.copy()

    if not metrics:
        zones["exposure_index"] = np.nan
        logger.warning("No metrics provided -- exposure_index set to NaN.")
        return zones

    # Filter to available columns
    available = {name: col for name, col in metrics.items() if col in zones.columns}
    if not available:
        zones["exposure_index"] = np.nan
        logger.warning(
            "None of the requested metric columns found -- exposure_index set to NaN."
        )
        return zones

    # Resolve weights
    if weights is None:
        weights = {name: 1.0 for name in available}
    else:
        weights = {name: weights.get(name, 0.0) for name in available}

    total_weight = sum(weights.values())
    if total_weight <= 0:
        zones["exposure_index"] = np.nan
        logger.warning("Total weight is zero -- exposure_index set to NaN.")
        return zones

    norm_weights = {name: w / total_weight for name, w in weights.items()}

    # Min-max normalise each metric and accumulate weighted sum
    weighted_sum = np.zeros(len(zones), dtype=float)
    valid_mask = np.ones(len(zones), dtype=bool)

    for name, col in available.items():
        values = zones[col].values.astype(float)
        nan_mask = np.isnan(values)
        valid_mask &= ~nan_mask

        finite = values[~nan_mask]
        if len(finite) == 0 or finite.max() == finite.min():
            # Cannot normalise -- treat as zero contribution
            normalised = np.zeros_like(values)
        else:
            vmin = finite.min()
            vmax = finite.max()
            normalised = (values - vmin) / (vmax - vmin)

        normalised = np.where(nan_mask, 0.0, normalised)
        weighted_sum += normalised * norm_weights[name]

    zones["exposure_index"] = weighted_sum
    zones.loc[~valid_mask, "exposure_index"] = np.nan

    logger.info(
        "Exposure index computed: %d valid zones, mean=%.3f.",
        valid_mask.sum(),
        np.nanmean(zones["exposure_index"].values),
    )
    return zones


# ---------------------------------------------------------------------------
# 5. Multi-panel choropleth
# ---------------------------------------------------------------------------

# Default panel layout: (column, colormap, title)
_EXPOSURE_PANEL_DEFAULTS = [
    ("solar_deficit", "OrRd", "Solar Deficit"),
    ("svf_deficit", "OrRd", "SVF Deficit"),
    ("density_ratio", "OrRd", "Volumetric Density"),
    ("exposure_index", "YlOrRd", "Composite Exposure Index"),
]

# Panel labels following Nature style
_PANEL_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"]


def plot_exposure_panel(
    zones_gdf: gpd.GeoDataFrame,
    output_path: Path,
    metrics_to_plot: list[tuple[str, str, str]] | None = None,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
) -> Path:
    """Multi-panel choropleth of exposure metrics.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Zone polygons with metric columns.
    output_path : Path
        Where to save the figure.
    metrics_to_plot : list of (column, cmap, title) tuples, optional
        Override the default panel layout.
    footprints_gdf : GeoDataFrame, optional
        Building footprints as light grey underlay.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary -- zones are clipped to this boundary.

    Returns
    -------
    Path
        *output_path* on success.
    """
    from src.cartography import apply_publication_style

    if metrics_to_plot is None:
        metrics_to_plot = _EXPOSURE_PANEL_DEFAULTS

    apply_publication_style()

    # Clip zones to boundary if provided
    plot_gdf = zones_gdf.copy()
    if boundary_gdf is not None:
        try:
            plot_gdf = gpd.clip(zones_gdf, boundary_gdf)
        except Exception:
            plot_gdf = zones_gdf.copy()

    # Drop zones with all-NaN metrics to avoid drawing empty cells
    metric_cols = [col for col, _, _ in metrics_to_plot if col in plot_gdf.columns]
    if metric_cols:
        has_any_data = plot_gdf[metric_cols].notna().any(axis=1)
        plot_gdf = plot_gdf.loc[has_any_data].copy()

    # Filter to available columns
    available = [
        (col, cmap, title)
        for col, cmap, title in metrics_to_plot
        if col in plot_gdf.columns
    ]
    if not available:
        logger.warning("No metric columns found for exposure panel -- skipping.")
        return output_path

    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (col, cmap, title) in enumerate(available):
        ax = axes[idx]

        # Building underlay (drawn first, below zones)
        if footprints_gdf is not None and len(footprints_gdf) > 0:
            footprints_gdf.plot(
                ax=ax,
                facecolor="#e0e0e0",
                edgecolor="#bbbbbb",
                linewidth=0.2,
                alpha=0.4,
                zorder=0,
            )

        # Only plot zones that have data for this column
        col_valid = plot_gdf[col].replace([np.inf, -np.inf], np.nan)
        valid_mask = col_valid.notna()
        plot_subset = plot_gdf.loc[valid_mask].copy()

        data = col_valid.dropna()
        if len(data) > 0:
            vmin = float(data.quantile(0.02))
            vmax = float(data.quantile(0.98))
            if vmin == vmax:
                vmax = vmin + 0.1
        else:
            vmin, vmax = 0.0, 1.0

        plot_subset.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={
                "label": title,
                "shrink": 0.65,
                "pad": 0.02,
                "orientation": "horizontal",
                "fraction": 0.04,
                "aspect": 30,
            },
            vmin=vmin,
            vmax=vmax,
            edgecolor="#888888",
            linewidth=0.3,
            alpha=0.85,
            zorder=2,
        )

        # Style the colorbar
        for child_ax in fig.axes:
            if child_ax not in axes and child_ax not in fig.axes[: len(axes)]:
                child_ax.tick_params(labelsize=7)

        if boundary_gdf is not None:
            add_settlement_boundary(ax, boundary_gdf)

        ax.set_title(title, fontsize=12, fontweight="bold")

        # Panel label (a, b, c, d)
        if idx < len(_PANEL_LABELS):
            ax.text(
                0.02,
                0.98,
                _PANEL_LABELS[idx],
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                va="top",
                ha="left",
                zorder=20,
            )

        add_scale_bar(ax)
        add_north_arrow(ax)
        ax.set_axis_off()

    for ax in axes[len(available) :]:
        ax.set_visible(False)

    fig.tight_layout(h_pad=2, w_pad=2)
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved exposure panel to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 6. Bivariate choropleth
# ---------------------------------------------------------------------------


def _build_bivariate_cmap(n: int = 3) -> tuple[ListedColormap, list[str]]:
    """Build an n x n bivariate colour grid.

    Rows = y-axis metric (low to high, bottom to top).
    Columns = x-axis metric (low to high, left to right).

    Returns (cmap, labels) where labels are 'row-col' strings.
    """
    # 3x3 palette inspired by Joshua Stevens bivariate schemes
    palette_3x3 = [
        # row 0 (low y): light teal  -> medium blue  -> dark blue
        "#e8e8e8",
        "#b5c0da",
        "#6c83b5",
        # row 1 (mid y): light pink  -> medium purple -> dark purple
        "#e4acac",
        "#ad9ea5",
        "#627f8c",
        # row 2 (high y): dark red   -> dark magenta  -> dark navy
        "#c85a5a",
        "#985356",
        "#574249",
    ]

    if n != 3:
        raise ValueError("Only n=3 is supported for the bivariate colormap.")

    cmap = ListedColormap(palette_3x3, name="bivariate_3x3")
    labels = [f"{r}-{c}" for r in range(n) for c in range(n)]
    return cmap, labels


def plot_exposure_bivariate(
    zones_gdf: gpd.GeoDataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    n_classes: int = 3,
) -> Path:
    """Bivariate choropleth mapping two exposure metrics simultaneously.

    Zones are classified into an *n_classes* x *n_classes* grid using
    quantile breaks on each metric.

    Parameters
    ----------
    zones_gdf : GeoDataFrame
        Must contain *x_col* and *y_col*.
    output_path : Path
        Where to save the figure.
    x_col, y_col : str
        Column names for the x- and y-axis metrics.
    footprints_gdf : GeoDataFrame, optional
        Building footprints underlay.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary.
    n_classes : int
        Number of classes per axis (default 3).

    Returns
    -------
    Path
        *output_path* on success.
    """
    if x_col not in zones_gdf.columns or y_col not in zones_gdf.columns:
        logger.warning(
            "Missing column(s) for bivariate plot: x=%s, y=%s -- skipping.",
            x_col,
            y_col,
        )
        return output_path

    gdf = zones_gdf.copy()

    # Clip
    if boundary_gdf is not None:
        try:
            gdf = gpd.clip(gdf, boundary_gdf)
            # Drop slivers created by clipping (< 10% of median zone area)
            areas = gdf.geometry.area
            median_area = areas.median()
            if median_area > 0:
                gdf = gdf.loc[areas >= median_area * 0.1].copy()
        except Exception:
            pass

    # Drop rows where either metric is NaN
    valid_mask = gdf[x_col].notna() & gdf[y_col].notna()
    gdf_valid = gdf.loc[valid_mask].copy()

    if len(gdf_valid) < n_classes:
        logger.warning(
            "Too few valid zones (%d) for bivariate plot -- skipping.", len(gdf_valid)
        )
        return output_path

    # Quantile classification
    x_vals = gdf_valid[x_col].values.astype(float)
    y_vals = gdf_valid[y_col].values.astype(float)

    x_breaks = np.quantile(x_vals, np.linspace(0, 1, n_classes + 1))
    y_breaks = np.quantile(y_vals, np.linspace(0, 1, n_classes + 1))

    # Ensure strictly increasing breaks
    x_breaks = np.unique(x_breaks)
    y_breaks = np.unique(y_breaks)

    x_class = np.clip(np.digitize(x_vals, x_breaks[1:-1]), 0, n_classes - 1)
    y_class = np.clip(np.digitize(y_vals, y_breaks[1:-1]), 0, n_classes - 1)

    # Bivariate class index: row * n_classes + col
    gdf_valid["_biv_class"] = y_class * n_classes + x_class

    cmap, labels = _build_bivariate_cmap(n_classes)
    colors = [cmap.colors[int(c)] for c in gdf_valid["_biv_class"].values]

    # --- Plot ---
    from src.cartography import apply_publication_style

    apply_publication_style()

    fig, (ax_map, ax_legend) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [5, 1]}
    )

    # Fill boundary as white background to mask empty areas
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.plot(
            ax=ax_map,
            facecolor="white",
            edgecolor="none",
            zorder=0,
        )

    # Building underlay
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        footprints_gdf.plot(
            ax=ax_map,
            facecolor="#e0e0e0",
            edgecolor="#bbbbbb",
            linewidth=0.2,
            alpha=0.4,
            zorder=1,
        )

    gdf_valid.plot(
        ax=ax_map, color=colors, edgecolor="#777777", linewidth=0.3, zorder=2
    )

    if boundary_gdf is not None:
        add_settlement_boundary(ax_map, boundary_gdf)

    # Human-readable column labels
    _col_labels = {
        "solar_deficit": "Solar Deficit",
        "svf_deficit": "SVF Deficit",
        "density_ratio": "Volumetric Density",
        "exposure_index": "Exposure Index",
        "mean_solar": "Mean Solar Hours",
        "mean_svf": "Mean SVF",
    }
    x_label = _col_labels.get(x_col, x_col)
    y_label = _col_labels.get(y_col, y_col)
    ax_map.set_title(f"{y_label}  vs  {x_label}", fontsize=12, fontweight="bold")
    add_scale_bar(ax_map)
    add_north_arrow(ax_map)
    ax_map.set_axis_off()

    # --- Legend: n x n colour grid ---
    legend_img = np.arange(n_classes * n_classes).reshape(n_classes, n_classes)
    ax_legend.imshow(
        legend_img,
        cmap=cmap,
        origin="lower",
        extent=[0, n_classes, 0, n_classes],
        interpolation="nearest",
    )
    ax_legend.set_xlabel(x_label, fontsize=9)
    ax_legend.set_ylabel(y_label, fontsize=9)
    ax_legend.set_xticks([0.5, n_classes - 0.5])
    ax_legend.set_xticklabels(["Low", "High"], fontsize=8)
    ax_legend.set_yticks([0.5, n_classes - 0.5])
    ax_legend.set_yticklabels(["Low", "High"], fontsize=8)
    ax_legend.set_title("Legend", fontsize=10, fontweight="bold")
    ax_legend.tick_params(length=0)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved bivariate exposure map to %s", output_path)
    return output_path

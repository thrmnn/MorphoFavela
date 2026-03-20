"""Reusable cartographic helpers for publication-quality spatial plots.

Every spatial map in the project should call these helpers to ensure
consistent cartographic elements: scale bar, north arrow, formatted axes,
terrain contours, settlement boundary, and building styling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from src.config import (
    BACKGROUND_COLOR,
    BUILDING_CMAP,
    CONTOUR_COLOR,
    CONTOUR_INTERVAL,
    SVF_CMAP,
)

logger = logging.getLogger(__name__)

# Scale bar length candidates (metres)
_SCALE_LENGTHS = [10, 20, 50, 100, 200, 500, 1000, 2000]


def add_scale_bar(
    ax: Axes,
    length_m: Optional[float] = None,
    location: str = "lower left",
) -> None:
    """Add a pure-matplotlib scale bar, auto-sized from axes extent.

    Parameters
    ----------
    ax : Axes
        Target matplotlib axes (must use projected CRS, i.e. metres).
    length_m : float, optional
        Explicit bar length in metres. If *None*, auto-selects from extent.
    location : str
        Corner placement: 'lower left', 'lower right', 'upper left', 'upper right'.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    extent_x = xlim[1] - xlim[0]
    extent_y = ylim[1] - ylim[0]

    if length_m is None:
        target = extent_x * 0.2
        length_m = min(_SCALE_LENGTHS, key=lambda v: abs(v - target))

    pad_x = extent_x * 0.03
    pad_y = extent_y * 0.03
    bar_height = extent_y * 0.008

    if "lower" in location:
        y0 = ylim[0] + pad_y
    else:
        y0 = ylim[1] - pad_y - bar_height

    if "left" in location:
        x0 = xlim[0] + pad_x
    else:
        x0 = xlim[1] - pad_x - length_m

    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            length_m,
            bar_height,
            facecolor="black",
            edgecolor="black",
            linewidth=0.5,
            zorder=10,
            clip_on=False,
        )
    )

    if length_m >= 1000:
        label = f"{length_m / 1000:.0f} km"
    else:
        label = f"{length_m:.0f} m"

    ax.text(
        x0 + length_m / 2,
        y0 + bar_height + extent_y * 0.008,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        zorder=10,
        clip_on=False,
    )


def add_north_arrow(
    ax: Axes,
    location: str = "upper right",
    size: float = 0.06,
) -> None:
    """Add a north arrow (triangle + 'N' label) via ax.annotate.

    Parameters
    ----------
    ax : Axes
        Target axes.
    location : str
        Corner placement.
    size : float
        Arrow size as fraction of axes height.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    extent_x = xlim[1] - xlim[0]
    extent_y = ylim[1] - ylim[0]
    arrow_len = extent_y * size
    pad = 0.04

    if "right" in location:
        cx = xlim[1] - extent_x * pad
    else:
        cx = xlim[0] + extent_x * pad

    if "upper" in location:
        base_y = ylim[1] - extent_y * pad - arrow_len
    else:
        base_y = ylim[0] + extent_y * pad

    ax.annotate(
        "",
        xy=(cx, base_y + arrow_len),
        xytext=(cx, base_y),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        zorder=10,
    )
    ax.text(
        cx,
        base_y + arrow_len + extent_y * 0.012,
        "N",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        zorder=10,
    )


def format_utm_axes(ax: Axes, show_labels: bool = True) -> None:
    """Format axes ticks for UTM coordinates.

    Parameters
    ----------
    ax : Axes
        Target axes.
    show_labels : bool
        If True, format ticks as '680.0 km E'. If False, hide labels.
    """
    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        return

    ax.ticklabel_format(useOffset=False, style="plain")

    def _fmt_easting(x, _pos):
        return f"{x / 1000:.1f} km E"

    def _fmt_northing(y, _pos):
        return f"{y / 1000:.1f} km N"

    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_easting))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_northing))
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")


def add_terrain_contours(
    ax: Axes,
    dtm_path: str | Path,
    interval_m: float = CONTOUR_INTERVAL,
    color: str = CONTOUR_COLOR,
    alpha: float = 0.3,
) -> None:
    """Overlay DTM contours via rasterio and ax.contour().

    Parameters
    ----------
    ax : Axes
        Target axes.
    dtm_path : str or Path
        Path to a GeoTIFF DTM raster.
    interval_m : float
        Contour interval in metres.
    color : str
        Contour line colour.
    alpha : float
        Contour transparency.
    """
    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio not available — skipping contours")
        return

    try:
        with rasterio.open(dtm_path) as src:
            dtm = src.read(1)
            bounds = src.bounds
    except Exception as e:
        logger.warning("Could not read DTM for contours: %s", e)
        return

    valid = dtm[np.isfinite(dtm) & (dtm > -1000)]
    if len(valid) == 0:
        return

    lo = np.floor(valid.min() / interval_m) * interval_m
    hi = np.ceil(valid.max() / interval_m) * interval_m + interval_m
    n_levels = int((hi - lo) / interval_m)
    if n_levels > 200:
        interval_m = max(interval_m, (hi - lo) / 50)
    levels = np.arange(lo, hi, interval_m)

    rows, cols = dtm.shape
    xs = np.linspace(bounds.left, bounds.right, cols)
    ys = np.linspace(bounds.top, bounds.bottom, rows)

    ax.contour(
        xs,
        ys,
        dtm,
        levels=levels,
        colors=color,
        linewidths=0.4,
        alpha=alpha,
        zorder=0,
    )


def add_settlement_boundary(
    ax: Axes,
    boundary_gdf: gpd.GeoDataFrame,
    edgecolor: str = "k",
    linestyle: str = "--",
) -> None:
    """Draw a dashed settlement boundary outline.

    Parameters
    ----------
    ax : Axes
        Target axes.
    boundary_gdf : GeoDataFrame
        Boundary polygon(s).
    edgecolor : str
        Outline colour.
    linestyle : str
        Line style.
    """
    boundary_gdf.boundary.plot(
        ax=ax,
        color=edgecolor,
        linestyle=linestyle,
        linewidth=1.2,
        zorder=9,
    )


def style_buildings_by_height(
    ax: Axes,
    gdf: gpd.GeoDataFrame,
    height_col: str = "altura",
    cmap: str = BUILDING_CMAP,
    alpha: float = 0.8,
) -> None:
    """Plot building footprints shaded by height.

    Falls back to flat light grey if *height_col* is absent.

    Parameters
    ----------
    ax : Axes
        Target axes.
    gdf : GeoDataFrame
        Building footprint polygons.
    height_col : str
        Column with building heights.
    cmap : str
        Colormap name.
    alpha : float
        Fill transparency.
    """
    if gdf is None or len(gdf) == 0:
        return

    if height_col in gdf.columns:
        valid = gdf[height_col].replace([np.inf, -np.inf], np.nan).dropna()
        vmin = valid.min() if len(valid) > 0 else 0
        vmax = valid.max() if len(valid) > 0 else 1
        gdf.plot(
            ax=ax,
            column=height_col,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            edgecolor="#555555",
            linewidth=0.3,
            zorder=1,
            legend=False,
        )
    else:
        gdf.plot(
            ax=ax,
            facecolor="#d0d0d0",
            edgecolor="#999999",
            linewidth=0.3,
            alpha=alpha,
            zorder=1,
        )


def get_svf_cmap() -> str:
    """Return the project-standard colorblind-safe SVF colormap name."""
    return SVF_CMAP


def apply_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality output.

    Call once at the top of a plotting script or in a figure setup block.
    Uses sans-serif fonts, outward ticks, and thin spines.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

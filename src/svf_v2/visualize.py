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

from src.config import DPI, FIGURE_SIZE

logger = logging.getLogger(__name__)


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
        footprints_gdf.plot(
            ax=ax,
            facecolor="#2d2d2d",
            edgecolor="#555555",
            linewidth=0.3,
            alpha=0.7,
            zorder=1,
        )

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=svf,
        cmap="RdYlGn",
        s=2,
        vmin=0,
        vmax=1,
        alpha=0.8,
        zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="SVF")
    ax.set_aspect("equal")
    ax.set_title("Sky View Factor (Grid)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_facecolor("#f0f0f0")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


def plot_street_svf(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
):
    """Street SVF scatter map with building footprints for spatial context."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Building footprints as dark context layer
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        footprints_gdf.plot(
            ax=ax,
            facecolor="#2d2d2d",
            edgecolor="#555555",
            linewidth=0.3,
            alpha=0.7,
            zorder=1,
        )

    # Road centrelines
    if roads_gdf is not None:
        roads_gdf.plot(ax=ax, color="#aaaaaa", linewidth=0.5, zorder=2)

    # SVF points on top
    gdf.plot(
        ax=ax,
        column="svf",
        cmap="RdYlGn",
        markersize=4,
        vmin=0,
        vmax=1,
        legend=True,
        legend_kwds={"label": "SVF", "shrink": 0.6},
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_title("Street SVF")
    ax.set_facecolor("#f0f0f0")
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
        cmap="RdYlGn",
        s=1,
        alpha=0.3,
        vmin=0,
        vmax=1,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Facade SVF by Orientation")
    plt.colorbar(sc, ax=ax, label="SVF", shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


def plot_facade_by_height(gdf: gpd.GeoDataFrame, output_path: Path):
    """Scatter of facade SVF vs height above ground."""
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
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", output_path)


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

    # ---- Top-left: SVF scatter map ----
    if roads_gdf is not None and not roads_gdf.empty:
        roads_gdf.plot(ax=ax_map, color="#cccccc", linewidth=0.5, zorder=1)

    street_gdf.plot(
        ax=ax_map,
        column="svf",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        markersize=3,
        legend=True,
        legend_kwds={"label": "SVF", "shrink": 0.6},
        zorder=2,
    )
    ax_map.set_title("Street SVF Map")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    ax_map.set_aspect("equal")

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

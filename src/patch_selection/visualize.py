"""Diagnostic visualizations for CFD patch selection.

Provides tile cluster maps, PCA scatter plots, feature distribution charts,
and DTM coverage maps to aid in evaluating and debugging the patch selection
pipeline.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

try:
    from src.config import DPI
except ImportError:
    DPI = 300

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Create parent directories for *path* if they don't exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_cluster_cmap(clusters: Sequence):
    """Return a {cluster_label: color} mapping using the tab10 palette."""
    labels = sorted(clusters)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(labels), 1))
    return {c: cmap(i) for i, c in enumerate(labels)}


# ---------------------------------------------------------------------------
# 1. Tile cluster map
# ---------------------------------------------------------------------------


def plot_tile_clusters(
    tiles_gdf: gpd.GeoDataFrame,
    selected_gdf: gpd.GeoDataFrame,
    output_path: Path,
    boundary: gpd.GeoDataFrame | None = None,
    buildings: gpd.GeoDataFrame | None = None,
) -> Path:
    """Map of all tiles colored by cluster with selected tiles highlighted.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        All candidate tiles.  Must contain a ``cluster`` column.
    selected_gdf : GeoDataFrame
        Subset of tiles chosen for CFD simulation.  Must contain a
        ``tile_id`` column for labelling.
    output_path : Path
        Destination PNG path.
    boundary : GeoDataFrame, optional
        Settlement boundary shown as a dashed black line.
    buildings : GeoDataFrame, optional
        Building footprints drawn as a light-gray underlay.

    Returns
    -------
    Path
        *output_path* for chaining.
    """
    output_path = Path(output_path)

    if "cluster" not in tiles_gdf.columns:
        logger.warning(
            "Column 'cluster' not found in tiles_gdf -- skipping tile cluster map."
        )
        return output_path

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Building footprint underlay
    if buildings is not None and len(buildings) > 0:
        buildings.plot(
            ax=ax,
            facecolor="#e0e0e0",
            edgecolor="#bbbbbb",
            linewidth=0.2,
            alpha=0.4,
            zorder=0,
        )

    # Cluster color map
    clusters = tiles_gdf["cluster"].dropna().unique()
    color_map = _build_cluster_cmap(clusters)

    # Identify selected tile indices for styling split
    selected_idx = set(selected_gdf.index) if selected_gdf is not None else set()

    # Non-selected tiles: thin gray outlines
    non_sel = tiles_gdf.loc[~tiles_gdf.index.isin(selected_idx)]
    if len(non_sel) > 0:
        colors = non_sel["cluster"].map(color_map).fillna("lightgray")
        non_sel.plot(
            ax=ax,
            color=colors.tolist(),
            edgecolor="gray",
            linewidth=0.4,
            alpha=0.7,
            zorder=1,
        )

    # Selected tiles: bold black outlines
    sel = tiles_gdf.loc[tiles_gdf.index.isin(selected_idx)]
    if len(sel) > 0:
        colors = sel["cluster"].map(color_map).fillna("lightgray")
        sel.plot(
            ax=ax,
            color=colors.tolist(),
            edgecolor="black",
            linewidth=2,
            alpha=0.9,
            zorder=2,
        )

        # Label selected tiles with tile_id
        if "tile_id" in selected_gdf.columns:
            for _, row in selected_gdf.iterrows():
                centroid = row.geometry.centroid
                label = str(row["tile_id"])
                ax.annotate(
                    label,
                    xy=(centroid.x, centroid.y),
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="black",
                    zorder=3,
                )

    # Boundary overlay
    if boundary is not None and len(boundary) > 0:
        boundary.boundary.plot(
            ax=ax,
            color="black",
            linestyle="--",
            linewidth=1.5,
            zorder=4,
        )

    # Legend
    patches = [
        Patch(facecolor=color_map[c], edgecolor="gray", label=f"Cluster {c}")
        for c in sorted(color_map)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    n_tiles = len(tiles_gdf)
    n_selected = len(selected_gdf) if selected_gdf is not None else 0
    n_clusters = len(clusters)
    ax.set_title(
        f"Tile Clusters ({n_tiles} tiles, {n_clusters} clusters, {n_selected} selected)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved tile cluster map to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. PCA scatter plot
# ---------------------------------------------------------------------------


def plot_pca_scatter(
    tiles_gdf: gpd.GeoDataFrame,
    selected_gdf: gpd.GeoDataFrame,
    output_path: Path,
    explained_variance: Sequence[float] | None = None,
) -> Path:
    """Scatter of PC1 vs PC2 with clusters and selected tiles highlighted.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        All tiles.  Must contain ``pc1``, ``pc2``, and ``cluster`` columns.
    selected_gdf : GeoDataFrame
        Selected tiles (drawn as star markers).  Must contain ``tile_id``.
    output_path : Path
        Destination PNG path.
    explained_variance : sequence of float, optional
        Explained variance ratio for each PC (e.g. ``[0.452, 0.231]``).
        Used to annotate axis labels.

    Returns
    -------
    Path
        *output_path* for chaining.
    """
    output_path = Path(output_path)

    required = {"pc1", "pc2", "cluster"}
    missing = required - set(tiles_gdf.columns)
    if missing:
        logger.warning(
            "Missing columns %s in tiles_gdf -- skipping PCA scatter.", missing
        )
        return output_path

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    clusters = tiles_gdf["cluster"].dropna().unique()
    color_map = _build_cluster_cmap(clusters)

    selected_idx = set(selected_gdf.index) if selected_gdf is not None else set()

    # Non-selected: small circles
    non_sel = tiles_gdf.loc[~tiles_gdf.index.isin(selected_idx)]
    for c in sorted(color_map):
        subset = non_sel.loc[non_sel["cluster"] == c]
        if len(subset) > 0:
            ax.scatter(
                subset["pc1"],
                subset["pc2"],
                c=[color_map[c]],
                s=30,
                alpha=0.6,
                edgecolors="gray",
                linewidth=0.3,
                label=f"Cluster {c}",
                zorder=1,
            )

    # Selected: star markers
    sel = tiles_gdf.loc[tiles_gdf.index.isin(selected_idx)]
    if len(sel) > 0:
        colors = sel["cluster"].map(color_map).fillna("lightgray")
        ax.scatter(
            sel["pc1"],
            sel["pc2"],
            c=colors.tolist(),
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=0.8,
            zorder=2,
            label="Selected",
        )

        # Annotate with tile_id
        if "tile_id" in selected_gdf.columns:
            for idx in sel.index:
                row = tiles_gdf.loc[idx]
                tid = (
                    selected_gdf.loc[idx, "tile_id"]
                    if idx in selected_gdf.index
                    else ""
                )
                ax.annotate(
                    str(tid),
                    xy=(row["pc1"], row["pc2"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    zorder=3,
                )

    # Axis labels
    xlabel = "PC1"
    ylabel = "PC2"
    if explained_variance is not None and len(explained_variance) >= 2:
        xlabel = f"PC1 ({explained_variance[0] * 100:.1f}%)"
        ylabel = f"PC2 ({explained_variance[1] * 100:.1f}%)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    ax.set_title("PCA Feature Space", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved PCA scatter plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Feature distributions (boxplots per cluster)
# ---------------------------------------------------------------------------


def plot_feature_distributions(
    tiles_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> Path:
    """Grid of boxplots showing feature distributions grouped by cluster.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        All tiles.  Must contain ``cluster`` and the requested feature columns.
    feature_columns : list[str]
        Column names to plot (one subplot per feature).
    output_path : Path
        Destination PNG path.

    Returns
    -------
    Path
        *output_path* for chaining.
    """
    output_path = Path(output_path)

    if "cluster" not in tiles_gdf.columns:
        logger.warning("Column 'cluster' not found -- skipping feature distributions.")
        return output_path

    # Filter to available columns
    available = [c for c in feature_columns if c in tiles_gdf.columns]
    if not available:
        logger.warning(
            "No requested feature columns found -- skipping feature distributions."
        )
        return output_path

    dropped = set(feature_columns) - set(available)
    if dropped:
        logger.warning("Feature columns not found, skipping: %s", dropped)

    n_features = len(available)
    ncols = min(3, n_features)
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    clusters = sorted(tiles_gdf["cluster"].dropna().unique())
    color_map = _build_cluster_cmap(clusters)

    # Try seaborn for nicer boxplots, fall back to matplotlib
    try:
        import seaborn as sns

        _use_seaborn = True
    except ImportError:
        _use_seaborn = False

    for idx, col in enumerate(available):
        ax = axes[idx]

        if _use_seaborn:
            sns.boxplot(
                data=tiles_gdf,
                x="cluster",
                y=col,
                order=clusters,
                palette=color_map,
                ax=ax,
                width=0.6,
            )
        else:
            # Manual matplotlib boxplot grouped by cluster
            data_by_cluster = [
                tiles_gdf.loc[tiles_gdf["cluster"] == c, col].dropna().values
                for c in clusters
            ]
            bp = ax.boxplot(
                data_by_cluster,
                labels=[str(c) for c in clusters],
                patch_artist=True,
                widths=0.6,
            )
            for patch, c in zip(bp["boxes"], clusters):
                patch.set_facecolor(color_map[c])
                patch.set_alpha(0.7)

        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    # Hide unused subplots
    for ax in axes[n_features:]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions by Cluster", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature distributions to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. DTM coverage map
# ---------------------------------------------------------------------------


def plot_dtm_coverage(
    tiles_gdf: gpd.GeoDataFrame,
    output_path: Path,
    boundary: gpd.GeoDataFrame | None = None,
) -> Path:
    """Map of tiles colored by DTM coverage quality.

    Looks for ``dtm_coverage_frac`` (continuous 0-1) first, then falls back
    to ``has_dtm_coverage`` (boolean).  Green indicates good coverage, red
    indicates poor coverage.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        All tiles.
    output_path : Path
        Destination PNG path.
    boundary : GeoDataFrame, optional
        Settlement boundary overlay.

    Returns
    -------
    Path
        *output_path* for chaining.
    """
    output_path = Path(output_path)

    # Choose which column to visualize
    if "dtm_coverage_frac" in tiles_gdf.columns:
        col = "dtm_coverage_frac"
        is_continuous = True
    elif "has_dtm_coverage" in tiles_gdf.columns:
        col = "has_dtm_coverage"
        is_continuous = False
    else:
        logger.warning(
            "Neither 'dtm_coverage_frac' nor 'has_dtm_coverage' found "
            "-- skipping DTM coverage map."
        )
        return output_path

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if is_continuous:
        # Continuous fraction: RdYlGn colormap (red=0, green=1)
        tiles_gdf.plot(
            column=col,
            ax=ax,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            legend=True,
            legend_kwds={
                "label": "DTM Coverage Fraction",
                "shrink": 0.65,
                "orientation": "horizontal",
                "fraction": 0.04,
                "pad": 0.06,
            },
            edgecolor="gray",
            linewidth=0.3,
            alpha=0.85,
            zorder=1,
        )
    else:
        # Boolean: simple green/red
        colors = tiles_gdf[col].map({True: "#4CAF50", False: "#F44336"})
        colors = colors.fillna("#999999")
        tiles_gdf.plot(
            ax=ax,
            color=colors.tolist(),
            edgecolor="gray",
            linewidth=0.3,
            alpha=0.85,
            zorder=1,
        )
        patches = [
            Patch(facecolor="#4CAF50", edgecolor="gray", label="Has DTM"),
            Patch(facecolor="#F44336", edgecolor="gray", label="No DTM"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=9)

    # Boundary overlay
    if boundary is not None and len(boundary) > 0:
        boundary.boundary.plot(
            ax=ax,
            color="black",
            linestyle="--",
            linewidth=1.5,
            zorder=4,
        )

    ax.set_title("DTM Coverage", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved DTM coverage map to %s", output_path)
    return output_path

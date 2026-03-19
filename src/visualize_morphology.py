"""Zone-level urban morphology visualizations.

Provides choropleths for zone metrics, LISA cluster maps, typology maps,
cluster profile charts, elbow/silhouette diagnostics, and Moran scatter plots.

Unlike ``visualize.py`` which focuses on per-building metrics (height, volume,
area), this module operates on grid-zone aggregates and categorical spatial
analysis outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import (
    COLORMAP_BCR,
    COLORMAP_FAR,
    COLORMAP_LAMBDA_F,
    COLORMAP_SIGMA_H,
    DPI,
    LISA_COLORS,
)

logger = logging.getLogger(__name__)

# Default metrics and their colormaps for the zone panel
_PANEL_DEFAULTS = [
    ("bcr", COLORMAP_BCR, "Building Coverage Ratio"),
    ("far", COLORMAP_FAR, "Floor Area Ratio"),
    ("sigma_h", COLORMAP_SIGMA_H, "Height Std Dev (m)"),
    ("lambda_f", COLORMAP_LAMBDA_F, "Frontal Area Index"),
]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Zone metrics panel
# ---------------------------------------------------------------------------


def plot_zone_metrics_panel(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    metrics: list[tuple[str, str, str]] | None = None,
) -> Path:
    """2x2 choropleth grid of zone-level morphology metrics.

    Parameters
    ----------
    gdf : GeoDataFrame
        Zone polygons with metric columns.
    output_path : Path
        Where to save the figure.
    metrics : list of (column, cmap, title) tuples, optional
        Override the default panel layout.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if metrics is None:
        metrics = _PANEL_DEFAULTS

    # Filter to available columns
    available = [(col, cmap, title) for col, cmap, title in metrics if col in gdf.columns]
    if not available:
        logger.warning("No metric columns found for zone panel — skipping.")
        return output_path

    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (col, cmap, title) in enumerate(available):
        ax = axes[idx]
        data = gdf[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 0:
            vmin = data.quantile(0.02)
            vmax = data.quantile(0.98)
        else:
            vmin, vmax = 0, 1
        gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={"label": col, "shrink": 0.7},
            vmin=vmin,
            vmax=vmax,
            missing_kwds={"color": "white", "edgecolor": "lightgray"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_axis_off()

    for ax in axes[len(available) :]:
        ax.set_visible(False)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved zone metrics panel to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. LISA cluster maps
# ---------------------------------------------------------------------------


def plot_lisa_clusters(
    gdf: gpd.GeoDataFrame,
    cluster_col: str,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Categorical choropleth of LISA cluster assignments.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *cluster_col* with values in {HH, HL, LH, LL, ns}.
    cluster_col : str
        Column name holding LISA cluster labels.
    output_path : Path
        Where to save the figure.
    title : str, optional
        Plot title (defaults to *cluster_col*).

    Returns
    -------
    Path
        *output_path* on success.
    """
    if cluster_col not in gdf.columns:
        logger.warning("Column '%s' not found — skipping LISA plot.", cluster_col)
        return output_path

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Map each zone to its LISA colour
    colors = gdf[cluster_col].map(LISA_COLORS).fillna(LISA_COLORS["ns"])
    gdf.plot(ax=ax, color=colors, edgecolor="gray", linewidth=0.2)

    # Legend
    present = gdf[cluster_col].unique()
    legend_order = ["HH", "HL", "LH", "LL", "ns"]
    patches = [
        Patch(facecolor=LISA_COLORS[lab], edgecolor="gray", label=lab)
        for lab in legend_order
        if lab in present
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    ax.set_title(title or cluster_col, fontsize=12, fontweight="bold")
    ax.set_axis_off()

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved LISA cluster map to %s", output_path)
    return output_path


def plot_lisa_clusters_panel(
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
) -> list[Path]:
    """Auto-discover ``lisa_cluster_*`` columns and plot each one.

    Returns
    -------
    list[Path]
        Paths to generated images.
    """
    cluster_cols = [c for c in gdf.columns if c.startswith("lisa_cluster_")]
    paths: list[Path] = []
    for col in cluster_cols:
        metric = col.replace("lisa_cluster_", "")
        out = output_dir / f"lisa_clusters_{metric}.png"
        plot_lisa_clusters(gdf, col, out, title=f"LISA Clusters — {metric}")
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# 3. Typology map
# ---------------------------------------------------------------------------


def plot_typology_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    cluster_col: str = "cluster",
) -> Path:
    """Categorical choropleth coloured by cluster label.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *cluster_col* (integer labels 0..n-1).
    output_path : Path
        Where to save the figure.
    cluster_col : str
        Column with cluster labels.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if cluster_col not in gdf.columns:
        logger.warning("Column '%s' not found — skipping typology map.", cluster_col)
        return output_path

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    clusters = sorted(gdf[cluster_col].dropna().unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(clusters), 1))
    color_map = {c: cmap(i) for i, c in enumerate(clusters)}
    colors = gdf[cluster_col].map(color_map).fillna("lightgray")

    gdf.plot(ax=ax, color=colors.tolist(), edgecolor="gray", linewidth=0.2)

    # Legend with zone counts
    counts = gdf[cluster_col].value_counts()
    patches = [
        Patch(facecolor=color_map[c], edgecolor="gray", label=f"Cluster {c} (n={counts.get(c, 0)})")
        for c in clusters
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.set_title("Settlement Typology Clusters", fontsize=12, fontweight="bold")
    ax.set_axis_off()

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved typology map to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Cluster profiles (grouped bar chart)
# ---------------------------------------------------------------------------


def plot_cluster_profiles(
    profiles_df: pd.DataFrame,
    output_path: Path,
    features: list[str] | None = None,
) -> Path:
    """Grouped bar chart of mean feature values per cluster.

    Parameters
    ----------
    profiles_df : DataFrame
        Indexed by cluster, with feature columns (+ ``count``).
    output_path : Path
        Where to save the figure.
    features : list[str], optional
        Subset of columns to plot.  Defaults to all non-count columns.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if features is None:
        features = [c for c in profiles_df.columns if c != "count"]
    features = [f for f in features if f in profiles_df.columns]
    if not features:
        logger.warning("No feature columns for cluster profiles — skipping.")
        return output_path

    n_clusters = len(profiles_df)
    n_features = len(features)
    x = np.arange(n_features)
    width = 0.8 / max(n_clusters, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_features * 1.5), 6))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

    for i, (cluster, row) in enumerate(profiles_df.iterrows()):
        count = row.get("count", "?")
        vals = [row[f] for f in features]
        offset = (i - n_clusters / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=f"Cluster {cluster} (n={count})", color=cmap(i))

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Cluster Profiles", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved cluster profiles to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 5. Elbow / silhouette diagnostic
# ---------------------------------------------------------------------------


def plot_elbow_silhouette(
    optimal_k_result: dict,
    output_path: Path,
) -> Path:
    """1x2 figure: elbow (inertia) and silhouette score vs k.

    Parameters
    ----------
    optimal_k_result : dict
        Output of ``find_optimal_k()`` with keys ``k_range``, ``inertia``,
        ``silhouette``, ``optimal_k``.
    output_path : Path
        Where to save the figure.

    Returns
    -------
    Path
        *output_path* on success.
    """
    k_range = optimal_k_result["k_range"]
    inertia = optimal_k_result["inertia"]
    silhouette = optimal_k_result["silhouette"]
    optimal_k = optimal_k_result["optimal_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow
    ax1.plot(k_range, inertia, "o-", color="steelblue", linewidth=2)
    ax1.axvline(optimal_k, color="red", linestyle="--", alpha=0.7)
    ax1.annotate(
        f"k={optimal_k}",
        xy=(optimal_k, inertia[k_range.index(optimal_k)]),
        xytext=(optimal_k + 0.5, inertia[k_range.index(optimal_k)]),
        fontsize=10,
        color="red",
    )
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Silhouette
    ax2.plot(k_range, silhouette, "o-", color="coral", linewidth=2)
    ax2.axvline(optimal_k, color="red", linestyle="--", alpha=0.7)
    ax2.annotate(
        f"k={optimal_k}",
        xy=(optimal_k, silhouette[k_range.index(optimal_k)]),
        xytext=(optimal_k + 0.5, silhouette[k_range.index(optimal_k)]),
        fontsize=10,
        color="red",
    )
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved elbow/silhouette plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 6. Moran scatter plot
# ---------------------------------------------------------------------------


def plot_moran_scatter(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    weights=None,
) -> Path:
    """Moran scatter plot (z vs Wz) with quadrant colouring.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *column*.
    column : str
        Numeric column to analyse.
    output_path : Path
        Where to save the figure.
    weights : libpysal.weights.W, optional
        Pre-computed weight matrix.  If *None*, queen contiguity weights are
        built internally.

    Returns
    -------
    Path
        *output_path* on success.
    """
    if column not in gdf.columns:
        logger.warning("Column '%s' not found — skipping Moran scatter.", column)
        return output_path

    from src.spatial_analysis import build_spatial_weights

    gdf = gdf.copy()
    valid_mask = gdf[column].notna()
    gdf_valid = gdf.loc[valid_mask].reset_index(drop=True)

    if len(gdf_valid) < 3:
        logger.warning("Too few valid observations for Moran scatter — skipping.")
        return output_path

    if weights is None:
        weights = build_spatial_weights(gdf_valid)

    y = gdf_valid[column].values.astype(float)
    mean_y = np.mean(y)
    std_y = np.std(y)
    if std_y == 0:
        logger.warning("Zero variance in '%s' — skipping Moran scatter.", column)
        return output_path

    z = (y - mean_y) / std_y

    # Spatial lag
    wz = np.zeros_like(z)
    for i in range(len(z)):
        nbrs = weights.neighbors[i]
        wts = weights.weights[i]
        if len(nbrs) > 0:
            wz[i] = sum(w * z[j] for j, w in zip(nbrs, wts))

    # Quadrant colours
    quadrant_colors = {
        "HH": LISA_COLORS["HH"],
        "HL": LISA_COLORS["HL"],
        "LH": LISA_COLORS["LH"],
        "LL": LISA_COLORS["LL"],
    }
    colors = []
    for zi, wzi in zip(z, wz):
        if zi >= 0 and wzi >= 0:
            colors.append(quadrant_colors["HH"])
        elif zi >= 0 and wzi < 0:
            colors.append(quadrant_colors["HL"])
        elif zi < 0 and wzi >= 0:
            colors.append(quadrant_colors["LH"])
        else:
            colors.append(quadrant_colors["LL"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z, wz, c=colors, s=15, alpha=0.6, edgecolors="gray", linewidth=0.3)

    # Regression line
    coeffs = np.polyfit(z, wz, 1)
    z_line = np.linspace(z.min(), z.max(), 100)
    ax.plot(z_line, np.polyval(coeffs, z_line), "k--", linewidth=1.5, alpha=0.7)

    # Axes
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"z ({column})", fontsize=11)
    ax.set_ylabel(f"Wz ({column})", fontsize=11)
    ax.set_title(f"Moran Scatter — {column} (slope={coeffs[0]:.3f})", fontsize=12, fontweight="bold")

    # Legend
    patches = [Patch(facecolor=v, edgecolor="gray", label=k) for k, v in quadrant_colors.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=9)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved Moran scatter to %s", output_path)
    return output_path

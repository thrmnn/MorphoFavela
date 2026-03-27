#!/usr/bin/env python3
"""
Cross-area clustering for CFD patch selection.

Loads tile features from multiple settlement areas, combines them into a single
dataset, performs global PCA + clustering, and selects representative patches
across all areas.  This enables a unified comparison of morphometric typologies
and ensures the final CFD patch set spans the full feature space.

Usage:
    python scripts/run_cross_area_clustering.py
    python scripts/run_cross_area_clustering.py --areas rocinha,maré --n-clusters 6
    python scripts/run_cross_area_clustering.py --n-clusters auto --output-dir outputs/comparative/cross_area_clustering
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DPI, OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cross_area_clustering")

# The 5 CFD campaign areas (CDD excluded).
CFD_AREAS = [
    "vidigal",
    "rocinha",
    "riodaspedras",
    "complexo_do_alemao",
    "maré",
]

# Columns that are metadata / geometry — not numeric features for clustering.
SKIP_COLS = {
    "tile_id",
    "area",
    "geometry",
    "geometry_buffered",
    "geometry_buffered_wkt",
    "classification",
    "boundary_overlap_frac",
    "n_buildings",
    "has_dtm_coverage",
    "buffer_distance_m",
    "cluster",
    "pc1",
    "pc2",
    "selection_reason",
    "dtm_coverage_frac",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_area_features(area: str) -> gpd.GeoDataFrame | None:
    """Load tiles_features.gpkg (or .csv fallback) for a single area.

    Returns None if no feature file is found.
    """
    base = OUTPUTS_DIR / area / "patch_selection"
    gpkg_path = base / "tiles_features.gpkg"
    csv_path = base / "tiles_features.csv"

    if gpkg_path.exists():
        gdf = gpd.read_file(gpkg_path)
        logger.info("Loaded %d tiles from %s (gpkg)", len(gdf), area)
        return gdf
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d tiles from %s (csv, no geometry)", len(df), area)
        return gpd.GeoDataFrame(df)
    else:
        logger.warning("No tiles_features found for %s — skipping.", area)
        return None


def load_all_areas(areas: list[str]) -> gpd.GeoDataFrame:
    """Load and combine tile features from all requested areas.

    Parameters
    ----------
    areas : list[str]
        Area names to load.

    Returns
    -------
    GeoDataFrame
        Combined tiles with an ``area`` column.

    Raises
    ------
    RuntimeError
        If fewer than 2 areas could be loaded.
    """
    frames: list[gpd.GeoDataFrame] = []
    for area in areas:
        gdf = _load_area_features(area)
        if gdf is not None and len(gdf) > 0:
            gdf = gdf.copy()
            gdf["area"] = area
            frames.append(gdf)

    if len(frames) < 2:
        loaded = [f["area"].iloc[0] for f in frames] if frames else []
        raise RuntimeError(
            f"Need at least 2 areas for cross-area clustering, but only "
            f"loaded: {loaded}. Check that tiles_features exist."
        )

    # Harmonize columns across frames — fill missing columns with NaN
    all_cols = set()
    for f in frames:
        all_cols.update(f.columns)
    for i, f in enumerate(frames):
        for col in all_cols - set(f.columns):
            frames[i][col] = np.nan

    combined = pd.concat(frames, ignore_index=True)
    # Preserve geometry if present
    if "geometry" in combined.columns:
        combined = gpd.GeoDataFrame(combined, geometry="geometry")

    logger.info(
        "Combined dataset: %d tiles from %d areas.",
        len(combined),
        len(frames),
    )
    return combined


# ---------------------------------------------------------------------------
# Feature matrix preparation
# ---------------------------------------------------------------------------


def _identify_feature_columns(
    combined: pd.DataFrame,
    min_nonnan_frac: float = 0.5,
) -> list[str]:
    """Identify numeric feature columns suitable for clustering.

    Drops metadata columns, and any column where more than
    ``(1 - min_nonnan_frac)`` of values are NaN (e.g. SVF columns when
    some areas lack SVF).

    Parameters
    ----------
    combined : DataFrame
        The combined tile dataset.
    min_nonnan_frac : float
        Minimum fraction of non-NaN values required to keep a column.

    Returns
    -------
    list[str]
        Sorted list of feature column names.
    """
    candidates = []
    n = len(combined)
    for col in combined.columns:
        if col in SKIP_COLS:
            continue
        if combined[col].dtype not in ("float64", "float32", "int64"):
            continue
        nonnan = combined[col].notna().sum()
        if nonnan / n >= min_nonnan_frac:
            candidates.append(col)
        else:
            logger.info(
                "Dropping feature '%s' — only %.0f%% non-NaN values.",
                col,
                100 * nonnan / n,
            )
    candidates.sort()
    return candidates


def _prepare_feature_matrix(
    combined: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Extract, impute (median), and z-score standardize the feature matrix."""
    X = combined[feature_cols].to_numpy(dtype=float)

    # Median imputation for residual NaNs
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col[mask] = median_val

    # Z-score standardization
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant columns
    X = (X - means) / stds

    return X


# ---------------------------------------------------------------------------
# PCA + Clustering (reuses logic from src.patch_selection.selection)
# ---------------------------------------------------------------------------


def _run_pca(
    X: np.ndarray,
    variance_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA dimensionality reduction — same logic as selection._run_pca."""
    from sklearn.decomposition import PCA

    n_max = min(X.shape[0], X.shape[1])
    pca_full = PCA(n_components=n_max, random_state=42)
    pca_full.fit(X)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, n_max)

    logger.info(
        "PCA: retaining %d / %d components (cum. variance=%.4f).",
        n_components,
        X.shape[1],
        cumvar[n_components - 1],
    )

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)
    return transformed, pca.components_, pca.explained_variance_ratio_


def _cluster(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster with k-medoids (sklearn_extra) or KMeans fallback."""
    try:
        from sklearn_extra.cluster import KMedoids

        logger.info("Using KMedoids (n_clusters=%d).", n_clusters)
        model = KMedoids(n_clusters=n_clusters, random_state=random_state, method="pam")
        labels = model.fit_predict(X)
        return labels.astype(int), model.medoid_indices_.astype(int)
    except ImportError:
        pass

    from sklearn.cluster import KMeans

    logger.info("Using KMeans fallback (n_clusters=%d).", n_clusters)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)

    medoid_indices = np.empty(n_clusters, dtype=int)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        dists = np.linalg.norm(X[members] - km.cluster_centers_[k], axis=1)
        medoid_indices[k] = members[np.argmin(dists)]

    return labels.astype(int), medoid_indices


def _find_optimal_k(
    X: np.ndarray,
    k_range: range,
    random_state: int = 42,
) -> dict[str, Any]:
    """Silhouette sweep to find optimal k."""
    from sklearn.metrics import silhouette_score

    valid_ks = [k for k in k_range if 2 <= k < X.shape[0]]
    if not valid_ks:
        return {"k_range": [2], "silhouette_scores": [0.0], "optimal_k": 2}

    scores: list[float] = []
    for k in valid_ks:
        labels, _ = _cluster(X, n_clusters=k, random_state=random_state)
        sil = float(silhouette_score(X, labels))
        scores.append(sil)
        logger.debug("k=%d silhouette=%.4f", k, sil)

    best_idx = int(np.argmax(scores))
    optimal_k = valid_ks[best_idx]
    logger.info("Optimal k=%d (silhouette=%.4f).", optimal_k, scores[best_idx])

    return {
        "k_range": valid_ks,
        "silhouette_scores": scores,
        "optimal_k": optimal_k,
    }


# ---------------------------------------------------------------------------
# Representative selection — cross-area variant
# ---------------------------------------------------------------------------


def _select_representatives(
    combined: gpd.GeoDataFrame,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
    pca_scores: np.ndarray,
) -> gpd.GeoDataFrame:
    """Select representative patches: medoids + PCA extremes + transition tiles.

    The transition criterion here is different from the single-area version:
    instead of spatial contiguity, we pick tiles that are closest to the
    cluster decision boundary in PCA space (tiles whose distance to their
    centroid vs. the next-nearest centroid is smallest).
    """
    n = len(combined)
    combined = combined.copy()
    combined["cluster"] = labels.astype(int)
    combined["pc1"] = pca_scores[:, 0]
    combined["pc2"] = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else 0.0

    reasons: dict[int, list[str]] = {}

    def _add(idx: int, reason: str) -> None:
        reasons.setdefault(idx, [])
        if reason not in reasons[idx]:
            reasons[idx].append(reason)

    # 1) Medoids
    for idx in medoid_indices:
        _add(int(idx), "medoid")

    # 2) PCA extremes (PC1 and PC2 min/max)
    pc1 = pca_scores[:, 0]
    _add(int(np.argmin(pc1)), "pca_extreme")
    _add(int(np.argmax(pc1)), "pca_extreme")
    if pca_scores.shape[1] >= 2:
        pc2 = pca_scores[:, 1]
        _add(int(np.argmin(pc2)), "pca_extreme")
        _add(int(np.argmax(pc2)), "pca_extreme")

    # 3) Transition tiles — closest to cluster decision boundaries in PCA space
    n_clusters = len(set(labels))
    # Compute cluster centroids in PCA space
    centroids = np.array(
        [pca_scores[labels == k].mean(axis=0) for k in range(n_clusters)]
    )
    # For each tile, find distance to own centroid vs next-nearest centroid
    max_transition = max(n_clusters, 4)
    margin_scores = np.empty(n)
    for i in range(n):
        own_k = labels[i]
        own_dist = np.linalg.norm(pca_scores[i] - centroids[own_k])
        other_dists = [
            np.linalg.norm(pca_scores[i] - centroids[k])
            for k in range(n_clusters)
            if k != own_k
        ]
        next_dist = min(other_dists) if other_dists else np.inf
        margin_scores[i] = next_dist - own_dist  # small = near boundary

    # Pick tiles with smallest margin (most ambiguous cluster assignment)
    boundary_candidates = np.argsort(margin_scores)
    n_added = 0
    for idx in boundary_candidates:
        idx = int(idx)
        if idx not in reasons and n_added < max_transition:
            _add(idx, "transition")
            n_added += 1
        if n_added >= max_transition:
            break

    # Assemble
    selected_indices = sorted(reasons.keys())
    if not selected_indices:
        result = combined.iloc[:0].copy()
        result["selection_reason"] = pd.Series(dtype=str)
        return result

    selected = combined.iloc[selected_indices].copy()
    selected["selection_reason"] = [", ".join(reasons[i]) for i in selected_indices]

    logger.info(
        "Selected %d representative tiles: %d medoid, %d pca_extreme, %d transition.",
        len(selected),
        sum(1 for r in reasons.values() if "medoid" in r),
        sum(1 for r in reasons.values() if "pca_extreme" in r),
        sum(1 for r in reasons.values() if "transition" in r),
    )
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

# Consistent area color palette
AREA_COLORS = {
    "vidigal": "#e41a1c",
    "rocinha": "#377eb8",
    "riodaspedras": "#4daf4a",
    "complexo_do_alemao": "#984ea3",
    "maré": "#ff7f00",
}


def _area_color(area: str) -> str:
    return AREA_COLORS.get(area, "#999999")


def _cluster_cmap(clusters):
    labels = sorted(set(clusters))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(labels), 1))
    return {c: cmap(i) for i, c in enumerate(labels)}


def plot_pca_by_area(
    combined: pd.DataFrame,
    output_path: Path,
    explained_variance: np.ndarray | None = None,
) -> Path:
    """PCA scatter colored by area."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for area in sorted(combined["area"].unique()):
        subset = combined[combined["area"] == area]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            c=_area_color(area),
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
            label=area,
        )

    xlabel, ylabel = "PC1", "PC2"
    if explained_variance is not None and len(explained_variance) >= 2:
        xlabel = f"PC1 ({explained_variance[0] * 100:.1f}%)"
        ylabel = f"PC2 ({explained_variance[1] * 100:.1f}%)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Global PCA — Colored by Area", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved PCA-by-area plot to %s", output_path)
    return output_path


def plot_pca_by_cluster(
    combined: pd.DataFrame,
    selected: pd.DataFrame | None,
    output_path: Path,
    explained_variance: np.ndarray | None = None,
) -> Path:
    """PCA scatter colored by cluster, with selected tiles as stars."""
    fig, ax = plt.subplots(figsize=(10, 8))

    clusters = sorted(combined["cluster"].unique())
    cmap = _cluster_cmap(clusters)

    for c in clusters:
        subset = combined[combined["cluster"] == c]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            c=[cmap[c]],
            s=40,
            alpha=0.6,
            edgecolors="gray",
            linewidth=0.3,
            label=f"Cluster {c}",
        )

    # Overlay selected patches
    if selected is not None and len(selected) > 0:
        colors = [cmap.get(c, "gray") for c in selected["cluster"]]
        ax.scatter(
            selected["pc1"],
            selected["pc2"],
            c=colors,
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
            label="Selected",
        )
        # Annotate selected with area + tile_id
        for _, row in selected.iterrows():
            label = f"{row['area']}:{row['tile_id']}"
            ax.annotate(
                label,
                xy=(row["pc1"], row["pc2"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                fontweight="bold",
            )

    xlabel, ylabel = "PC1", "PC2"
    if explained_variance is not None and len(explained_variance) >= 2:
        xlabel = f"PC1 ({explained_variance[0] * 100:.1f}%)"
        ylabel = f"PC2 ({explained_variance[1] * 100:.1f}%)"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Global PCA — Colored by Cluster", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved PCA-by-cluster plot to %s", output_path)
    return output_path


def plot_feature_distributions_by_area(
    combined: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
) -> Path:
    """Boxplots of feature distributions grouped by area."""
    # Pick a representative subset of features to keep the plot readable
    n_features = len(feature_cols)
    ncols = min(4, n_features)
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    areas = sorted(combined["area"].unique())

    try:
        import seaborn as sns

        _use_seaborn = True
    except ImportError:
        _use_seaborn = False

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        if _use_seaborn:
            palette = {a: _area_color(a) for a in areas}
            sns.boxplot(
                data=combined,
                x="area",
                y=col,
                order=areas,
                palette=palette,
                ax=ax,
                width=0.6,
            )
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", fontsize=8
            )
        else:
            data_by_area = [
                combined.loc[combined["area"] == a, col].dropna().values for a in areas
            ]
            bp = ax.boxplot(
                data_by_area,
                tick_labels=[a[:8] for a in areas],
                patch_artist=True,
                widths=0.6,
            )
            for patch, a in zip(bp["boxes"], areas):
                patch.set_facecolor(_area_color(a))
                patch.set_alpha(0.7)
            ax.tick_params(axis="x", rotation=30, labelsize=8)

        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes[n_features:]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions by Area", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature distributions to %s", output_path)
    return output_path


def plot_selected_patches_map(
    combined: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    output_path: Path,
) -> Path:
    """Per-area maps showing selected patches highlighted."""
    has_geom = (
        "geometry" in combined.columns
        and combined.geometry is not None
        and not combined.geometry.isna().all()
    )
    if not has_geom:
        logger.info("No geometry available — skipping selected patches map.")
        return output_path

    areas = sorted(combined["area"].unique())
    n_areas = len(areas)
    ncols = min(3, n_areas)
    nrows = math.ceil(n_areas / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    clusters = sorted(combined["cluster"].unique())
    cmap = _cluster_cmap(clusters)

    for idx, area in enumerate(areas):
        ax = axes[idx]
        area_tiles = combined[combined["area"] == area]
        area_selected = (
            selected[selected["area"] == area] if selected is not None else None
        )

        # All tiles in area
        if "cluster" in area_tiles.columns:
            colors = area_tiles["cluster"].map(cmap).fillna("lightgray")
            area_tiles.plot(
                ax=ax,
                color=colors.tolist(),
                edgecolor="gray",
                linewidth=0.4,
                alpha=0.6,
            )
        else:
            area_tiles.plot(ax=ax, color="lightblue", edgecolor="gray", linewidth=0.4)

        # Selected tiles with bold outline
        if area_selected is not None and len(area_selected) > 0:
            colors_sel = area_selected["cluster"].map(cmap).fillna("lightgray")
            area_selected.plot(
                ax=ax,
                color=colors_sel.tolist(),
                edgecolor="black",
                linewidth=2.5,
                alpha=0.9,
            )
            # Label
            for _, row in area_selected.iterrows():
                if hasattr(row.geometry, "centroid"):
                    c = row.geometry.centroid
                    ax.annotate(
                        str(row["tile_id"]),
                        xy=(c.x, c.y),
                        fontsize=7,
                        fontweight="bold",
                        ha="center",
                        va="center",
                    )

        ax.set_title(
            f"{area} ({len(area_tiles)} tiles)", fontsize=11, fontweight="bold"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)", fontsize=8)
        ax.set_ylabel("Northing (m)", fontsize=8)

    # Legend
    patches = [
        Patch(facecolor=cmap[c], edgecolor="gray", label=f"Cluster {c}")
        for c in sorted(cmap)
    ]
    if nrows * ncols > n_areas:
        # Use the last unused axis for the legend
        legend_ax = axes[n_areas]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        legend_ax.legend(handles=patches, loc="center", fontsize=10, ncol=2)
    else:
        fig.legend(
            handles=patches,
            loc="lower center",
            fontsize=9,
            ncol=min(len(clusters), 6),
            bbox_to_anchor=(0.5, -0.02),
        )

    # Hide unused axes
    for ax in axes[n_areas + 1 :]:
        ax.set_visible(False)

    fig.suptitle("Selected Patches by Area", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved selected patches map to %s", output_path)
    return output_path


def plot_silhouette_curve(
    sil_info: dict,
    output_path: Path,
) -> Path:
    """Plot silhouette score vs. k for the sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sil_info["k_range"]
    scores = sil_info["silhouette_scores"]
    optimal = sil_info["optimal_k"]

    ax.plot(ks, scores, "o-", color="#377eb8", linewidth=2, markersize=6)
    ax.axvline(
        optimal,
        color="#e41a1c",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal k={optimal}",
    )
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved silhouette curve to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_composition_table(
    combined: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Print and return a cluster composition table (tiles per area per cluster)."""
    table = pd.crosstab(
        combined["area"],
        combined["cluster"],
        margins=True,
        margins_name="Total",
    )
    print("\n" + "=" * 70)
    print("CLUSTER COMPOSITION (tiles per area per cluster)")
    print("=" * 70)
    print(table.to_string())

    if selected is not None and len(selected) > 0:
        sel_table = pd.crosstab(
            selected["area"],
            selected["cluster"],
            margins=True,
            margins_name="Total",
        )
        print("\n" + "-" * 70)
        print("SELECTED PATCHES per area per cluster")
        print("-" * 70)
        print(sel_table.to_string())

    print("=" * 70 + "\n")
    return table


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def _json_convert(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, range):
        return list(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_cross_area_clustering(
    areas: list[str],
    n_clusters: int | None = None,
    k_range: range = range(4, 13),
    output_dir: Path = OUTPUTS_DIR / "comparative" / "cross_area_clustering",
    random_state: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    """Full cross-area clustering pipeline.

    Parameters
    ----------
    areas : list[str]
        Area names to include.
    n_clusters : int or None
        Fixed k, or None for silhouette-based auto-detection.
    k_range : range
        Candidate k values for auto-detection.
    output_dir : Path
        Output directory for results and visualizations.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        ``(combined_gdf, selected_gdf, metadata)``
    """
    t_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & combine ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: LOAD & COMBINE TILE FEATURES")
    logger.info("=" * 60)
    combined = load_all_areas(areas)

    # ── 2. Identify features & build matrix ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE MATRIX PREPARATION")
    logger.info("=" * 60)
    feature_cols = _identify_feature_columns(combined)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    X = _prepare_feature_matrix(combined, feature_cols)
    logger.info("Feature matrix shape: %s", X.shape)

    # ── 3. PCA ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: GLOBAL PCA")
    logger.info("=" * 60)
    pca_scores, pca_loadings, pca_evr = _run_pca(X)

    combined["pc1"] = pca_scores[:, 0]
    combined["pc2"] = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else 0.0

    # ── 4. Determine k ────────────────────────────────────────────────────
    metadata: dict[str, Any] = {}

    if n_clusters is None:
        logger.info("=" * 60)
        logger.info(
            "STEP 4a: SILHOUETTE SWEEP (k=%d..%d)", k_range.start, k_range.stop - 1
        )
        logger.info("=" * 60)
        sil_info = _find_optimal_k(
            pca_scores, k_range=k_range, random_state=random_state
        )
        n_clusters = sil_info["optimal_k"]
        metadata["silhouette_info"] = sil_info

        # Plot silhouette curve
        plot_silhouette_curve(sil_info, output_dir / "silhouette_curve.png")

    # Guard against too many clusters
    if n_clusters >= len(combined):
        n_clusters = max(2, len(combined) - 1)
        logger.warning("Reduced k to %d (fewer tiles than clusters).", n_clusters)

    # ── 5. Cluster ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: GLOBAL CLUSTERING (k=%d)", n_clusters)
    logger.info("=" * 60)
    labels, medoid_indices = _cluster(
        pca_scores, n_clusters=n_clusters, random_state=random_state
    )
    combined["cluster"] = labels.astype(int)

    # ── 6. Select representatives ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: REPRESENTATIVE SELECTION")
    logger.info("=" * 60)
    selected = _select_representatives(combined, labels, medoid_indices, pca_scores)

    # ── 7. Summary ────────────────────────────────────────────────────────
    composition = _print_composition_table(combined, selected)

    # ── 8. Metadata ───────────────────────────────────────────────────────
    # Compute cluster profiles on original (non-standardized) features
    from src.typology import compute_cluster_profiles

    cluster_profiles = compute_cluster_profiles(
        combined, feature_cols, cluster_col="cluster"
    )
    metadata.update(
        {
            "n_clusters": n_clusters,
            "n_tiles": len(combined),
            "n_selected": len(selected),
            "areas": sorted(combined["area"].unique().tolist()),
            "tiles_per_area": combined["area"].value_counts().to_dict(),
            "feature_columns": feature_cols,
            "pca_explained_variance": pca_evr.tolist(),
            "pca_n_components": pca_scores.shape[1],
            "pca_loadings": pca_loadings.tolist(),
            "cluster_labels": labels.tolist(),
            "cluster_profiles": (
                cluster_profiles.to_dict()
                if isinstance(cluster_profiles, pd.DataFrame)
                else cluster_profiles
            ),
            "composition": composition.to_dict(),
            "elapsed_seconds": round(time.time() - t_start, 1),
        }
    )

    # ── 9. Save outputs ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: SAVE OUTPUTS")
    logger.info("=" * 60)

    # Combined tiles with cluster assignments
    save_combined = combined.copy()
    if "geometry_buffered" in save_combined.columns:
        save_combined = save_combined.drop(columns=["geometry_buffered"])
    if "geometry_buffered_wkt" in save_combined.columns:
        save_combined = save_combined.drop(columns=["geometry_buffered_wkt"])

    has_geom = (
        "geometry" in save_combined.columns
        and save_combined.geometry is not None
        and not save_combined.geometry.isna().all()
    )
    if has_geom:
        save_combined.to_file(output_dir / "combined_tiles.gpkg", driver="GPKG")
    save_combined.drop(columns=["geometry"], errors="ignore").to_csv(
        output_dir / "combined_tiles.csv", index=False
    )
    logger.info("Saved combined tiles (%d rows).", len(save_combined))

    # Selected patches
    save_selected = selected.copy()
    if "geometry_buffered" in save_selected.columns:
        save_selected = save_selected.drop(columns=["geometry_buffered"])
    if "geometry_buffered_wkt" in save_selected.columns:
        save_selected = save_selected.drop(columns=["geometry_buffered_wkt"])

    has_geom_sel = (
        "geometry" in save_selected.columns
        and save_selected.geometry is not None
        and not save_selected.geometry.isna().all()
    )
    if has_geom_sel:
        save_selected.to_file(output_dir / "selected_patches.gpkg", driver="GPKG")
    save_selected.drop(columns=["geometry"], errors="ignore").to_csv(
        output_dir / "selected_patches.csv", index=False
    )
    logger.info("Saved %d selected patches.", len(save_selected))

    # Metadata JSON
    with open(output_dir / "cross_area_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_json_convert)
    logger.info("Saved metadata.")

    # Composition table CSV
    composition.to_csv(output_dir / "cluster_composition.csv")

    # ── 10. Visualizations ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: VISUALIZATIONS")
    logger.info("=" * 60)

    try:
        plot_pca_by_area(
            combined,
            output_dir / "pca_by_area.png",
            explained_variance=pca_evr,
        )
    except Exception as e:
        logger.warning("PCA-by-area plot failed: %s", e, exc_info=True)

    try:
        plot_pca_by_cluster(
            combined,
            selected,
            output_dir / "pca_by_cluster.png",
            explained_variance=pca_evr,
        )
    except Exception as e:
        logger.warning("PCA-by-cluster plot failed: %s", e, exc_info=True)

    try:
        plot_feature_distributions_by_area(
            combined,
            feature_cols,
            output_dir / "feature_distributions_by_area.png",
        )
    except Exception as e:
        logger.warning("Feature distributions plot failed: %s", e, exc_info=True)

    try:
        plot_selected_patches_map(
            combined,
            selected,
            output_dir / "selected_patches_map.png",
        )
    except Exception as e:
        logger.warning("Selected patches map failed: %s", e, exc_info=True)

    elapsed = time.time() - t_start
    logger.info(
        "Cross-area clustering complete: %d tiles, %d clusters, "
        "%d selected patches across %d areas (%.1fs).",
        len(combined),
        n_clusters,
        len(selected),
        len(combined["area"].unique()),
        elapsed,
    )

    return combined, selected, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-area clustering for CFD patch selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_cross_area_clustering.py
  python scripts/run_cross_area_clustering.py --areas rocinha,maré,riodaspedras
  python scripts/run_cross_area_clustering.py --n-clusters 8
  python scripts/run_cross_area_clustering.py --n-clusters auto --k-min 4 --k-max 12
""",
    )
    parser.add_argument(
        "--areas",
        type=str,
        default=",".join(CFD_AREAS),
        help=(f"Comma-separated area names (default: {','.join(CFD_AREAS)})"),
    )
    parser.add_argument(
        "--n-clusters",
        type=str,
        default="auto",
        help="Number of clusters: integer or 'auto' for silhouette sweep (default: auto)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=4,
        help="Minimum k for silhouette sweep (default: 4)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=12,
        help="Maximum k for silhouette sweep (default: 12)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_DIR / "comparative" / "cross_area_clustering"),
        help="Output directory (default: outputs/comparative/cross_area_clustering/)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Parse areas
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]

    # Parse n_clusters
    if args.n_clusters.lower() == "auto":
        n_clusters = None
    else:
        try:
            n_clusters = int(args.n_clusters)
        except ValueError:
            parser.error(
                f"--n-clusters must be an integer or 'auto', got: {args.n_clusters}"
            )

    k_range = range(args.k_min, args.k_max + 1)

    combined, selected, metadata = run_cross_area_clustering(
        areas=areas,
        n_clusters=n_clusters,
        k_range=k_range,
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
    )

    # Final summary
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Combined tiles:    {len(combined)}")
    print(f"  Clusters:          {metadata['n_clusters']}")
    print(f"  Selected patches:  {len(selected)}")
    print(f"  Areas included:    {', '.join(metadata['areas'])}")
    print(f"  Elapsed:           {metadata['elapsed_seconds']}s")

    # List selected patches
    print("\nSelected patches:")
    for _, row in selected.iterrows():
        print(
            f"  {row['area']:>20s} | tile {row['tile_id']:>6s} | "
            f"cluster {row['cluster']} | {row['selection_reason']}"
        )


if __name__ == "__main__":
    main()

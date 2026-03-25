"""Representative patch selection via PCA dimensionality reduction and clustering.

Reduces a tile feature matrix with PCA, clusters tiles using k-medoids (with
KMeans fallback), and selects a representative subset — medoids, PCA extremes,
and transition-zone tiles — for downstream CFD simulation.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.patch_selection.features import CLUSTERING_EXCLUDE, get_clustering_features
from src.spatial_analysis import build_spatial_weights
from src.typology import compute_cluster_profiles, normalize_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_sklearn(module_path: str) -> Any:
    """Lazy-import a scikit-learn submodule, raising a clear error if missing."""
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for patch selection but is not installed. "
            "Install it with:  pip install scikit-learn>=1.3.0"
        ) from exc


def _impute_median(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with the column median (in-place on a copy)."""
    arr = arr.copy()
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col[mask] = median_val
    return arr


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def _run_pca(
    X: np.ndarray,
    variance_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce dimensionality with PCA, keeping enough components for *variance_threshold*.

    Parameters
    ----------
    X : np.ndarray
        Standardised feature matrix of shape (n_samples, n_features).
    variance_threshold : float
        Minimum cumulative explained variance ratio to retain (default 0.90).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(transformed, loadings, explained_variance_ratio)`` where
        *transformed* has shape (n_samples, n_components), *loadings* has
        shape (n_components, n_features), and *explained_variance_ratio* is
        a 1-D array of per-component variance ratios.
    """
    decomp = _require_sklearn("sklearn.decomposition")
    PCA = decomp.PCA

    # Fit full PCA first to determine how many components are needed.
    n_components_max = min(X.shape[0], X.shape[1])
    pca_full = PCA(n_components=n_components_max, random_state=42)
    pca_full.fit(X)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    # Number of components to reach the threshold (at least 1).
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, n_components_max)

    logger.info(
        "PCA: retaining %d / %d components (cumulative variance=%.4f, threshold=%.2f).",
        n_components,
        X.shape[1],
        cumvar[n_components - 1],
        variance_threshold,
    )

    # Re-fit with the chosen number of components for a clean object.
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)

    return transformed, pca.components_, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster_kmedoids(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster with k-medoids, falling back to KMeans + nearest-to-centroid.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(labels, medoid_indices)`` — integer cluster labels and the row
        index of each cluster's medoid (shape ``(n_clusters,)``).
    """
    try:
        from sklearn_extra.cluster import KMedoids

        logger.info(
            "Using sklearn_extra KMedoids (n_clusters=%d).", n_clusters
        )
        model = KMedoids(
            n_clusters=n_clusters, random_state=random_state, method="pam"
        )
        labels = model.fit_predict(X)
        medoid_indices = model.medoid_indices_
        return labels.astype(int), medoid_indices.astype(int)
    except ImportError:
        logger.info(
            "sklearn_extra not available — falling back to KMeans + "
            "nearest-to-centroid medoid approximation (n_clusters=%d).",
            n_clusters,
        )

    cluster_mod = _require_sklearn("sklearn.cluster")
    KMeans = cluster_mod.KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)

    # Approximate medoids: for each cluster, find the sample nearest to centroid.
    medoid_indices = np.empty(n_clusters, dtype=int)
    for k in range(n_clusters):
        member_mask = labels == k
        member_indices = np.where(member_mask)[0]
        centroid = km.cluster_centers_[k]
        dists = np.linalg.norm(X[member_indices] - centroid, axis=1)
        medoid_indices[k] = member_indices[np.argmin(dists)]

    return labels.astype(int), medoid_indices


def _find_optimal_k(
    X: np.ndarray,
    k_range: range = range(4, 16),
    random_state: int = 42,
) -> dict[str, Any]:
    """Sweep *k* values and select the one with the highest silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    k_range : range
        Candidate cluster counts to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"k_range": list[int], "silhouette_scores": list[float], "optimal_k": int}``
    """
    metrics_mod = _require_sklearn("sklearn.metrics")
    silhouette_score = metrics_mod.silhouette_score

    n_samples = X.shape[0]
    # Silhouette requires k >= 2 and k < n_samples.
    valid_ks = [k for k in k_range if 2 <= k < n_samples]

    if not valid_ks:
        logger.warning(
            "No valid k in range %s for %d samples; defaulting to k=2.",
            k_range,
            n_samples,
        )
        return {"k_range": [2], "silhouette_scores": [0.0], "optimal_k": 2}

    silhouette_scores: list[float] = []

    for k in valid_ks:
        labels, _ = _cluster_kmedoids(X, n_clusters=k, random_state=random_state)
        sil = float(silhouette_score(X, labels))
        silhouette_scores.append(sil)
        logger.debug("k=%d  silhouette=%.4f", k, sil)

    best_idx = int(np.argmax(silhouette_scores))
    optimal_k = valid_ks[best_idx]
    logger.info(
        "Optimal k=%d (silhouette=%.4f).", optimal_k, silhouette_scores[best_idx]
    )

    return {
        "k_range": valid_ks,
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
    }


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------


def _select_representatives(
    tiles_gdf: gpd.GeoDataFrame,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
    pca_scores: np.ndarray,
) -> gpd.GeoDataFrame:
    """Select representative tiles: medoids, PCA extremes, and transition tiles.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        The full set of tiles (same row order as *labels* / *pca_scores*).
    labels : np.ndarray
        Integer cluster labels for each tile.
    medoid_indices : np.ndarray
        Row indices of cluster medoids.
    pca_scores : np.ndarray
        PCA-transformed coordinates, shape (n_tiles, n_components).  At least
        two columns (PC1, PC2) are expected.

    Returns
    -------
    GeoDataFrame
        Subset of *tiles_gdf* with added columns: ``cluster``, ``pc1``,
        ``pc2``, ``selection_reason``.
    """
    n_tiles = len(tiles_gdf)

    # -- Annotate all tiles with cluster and PC scores -------------------
    tiles_gdf = tiles_gdf.copy()
    tiles_gdf["cluster"] = labels.astype(int)
    tiles_gdf["pc1"] = pca_scores[:, 0]
    tiles_gdf["pc2"] = pca_scores[:, 1] if pca_scores.shape[1] >= 2 else 0.0

    # Track selection reasons per row index.
    reasons: dict[int, list[str]] = {}

    def _add(idx: int, reason: str) -> None:
        reasons.setdefault(idx, [])
        if reason not in reasons[idx]:
            reasons[idx].append(reason)

    # 1) Medoids ----------------------------------------------------------
    for idx in medoid_indices:
        _add(int(idx), "medoid")

    # 2) PCA extremes (PC1 min/max, PC2 min/max) -------------------------
    pc1 = pca_scores[:, 0]
    _add(int(np.argmin(pc1)), "pca_extreme")
    _add(int(np.argmax(pc1)), "pca_extreme")

    if pca_scores.shape[1] >= 2:
        pc2 = pca_scores[:, 1]
        _add(int(np.argmin(pc2)), "pca_extreme")
        _add(int(np.argmax(pc2)), "pca_extreme")

    # 3) Transition zone tiles --------------------------------------------
    #    Select up to 1 transition tile per cluster-pair boundary (capped to
    #    avoid selecting nearly every tile in a dense grid).
    k = len(set(labels))
    max_transition = max(k, 4)  # reasonable cap
    try:
        _gdf_for_weights = tiles_gdf[["geometry"]].copy()
        _gdf_for_weights = gpd.GeoDataFrame(_gdf_for_weights, geometry="geometry")
        w = build_spatial_weights(_gdf_for_weights, method="queen")

        seen_pairs: set[tuple[int, int]] = set()
        for i in range(n_tiles):
            if len(seen_pairs) >= max_transition:
                break
            tile_cluster = labels[i]
            for j in w.neighbors.get(i, []):
                if labels[j] != tile_cluster:
                    pair = tuple(sorted((int(tile_cluster), int(labels[j]))))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        _add(i, "transition")
                    break
    except Exception:
        logger.warning(
            "Could not build queen contiguity weights for transition "
            "detection — skipping transition tiles.",
            exc_info=True,
        )

    # -- Assemble output --------------------------------------------------
    selected_indices = sorted(reasons.keys())
    if not selected_indices:
        logger.warning("No representatives selected — returning empty GeoDataFrame.")
        result = tiles_gdf.iloc[:0].copy()
        result["selection_reason"] = pd.Series(dtype=str)
        return result

    selected = tiles_gdf.iloc[selected_indices].copy()
    selected["selection_reason"] = [
        ", ".join(reasons[i]) for i in selected_indices
    ]

    logger.info(
        "Selected %d representative tiles (%d medoid, %d pca_extreme, "
        "%d transition) from %d total.",
        len(selected),
        sum(1 for r in reasons.values() if "medoid" in r),
        sum(1 for r in reasons.values() if "pca_extreme" in r),
        sum(1 for r in reasons.values() if "transition" in r),
        n_tiles,
    )
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_patches(
    tiles_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    n_clusters: int | None = None,
    k_range: range = range(4, 16),
    random_state: int = 42,
    auto_exclude: bool = True,
    min_building_count: int = 0,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Full pipeline: standardise, reduce, cluster, and select representative tiles.

    Parameters
    ----------
    tiles_gdf : GeoDataFrame
        Tiles with morphometric feature columns.
    feature_columns : list[str]
        Column names to use as clustering features.  When *auto_exclude* is
        True, features listed in :data:`CLUSTERING_EXCLUDE` are automatically
        removed before PCA/clustering but are still present in the output.
    n_clusters : int or None
        Fixed number of clusters.  If *None*, the optimal *k* is auto-detected
        via silhouette sweep over *k_range*.
    k_range : range
        Candidate cluster counts when *n_clusters* is None.
    random_state : int
        Random seed for reproducibility.
    auto_exclude : bool
        If True (default), apply :data:`CLUSTERING_EXCLUDE` to filter out
        redundant / noisy / data-quality features before PCA.
    min_building_count : int
        If > 0, tiles with fewer than this many buildings (column
        ``n_buildings``) are dropped before clustering.

    Returns
    -------
    tuple[GeoDataFrame, dict]
        ``(selected_gdf, metadata)`` where *selected_gdf* contains the
        representative tiles with columns ``cluster``, ``pc1``, ``pc2``,
        ``selection_reason``, and *metadata* is a dict with keys:
        ``n_clusters``, ``pca_explained_variance``, ``pca_loadings``,
        ``silhouette_info`` (if auto-detected), ``cluster_profiles``.
    """
    # -- 0. Optional building-count filter --------------------------------
    if min_building_count > 0 and "n_buildings" in tiles_gdf.columns:
        n_before = len(tiles_gdf)
        tiles_gdf = tiles_gdf[tiles_gdf["n_buildings"] >= min_building_count].copy()
        tiles_gdf = tiles_gdf.reset_index(drop=True)
        n_dropped = n_before - len(tiles_gdf)
        if n_dropped > 0:
            logger.info(
                "Pre-clustering filter: dropped %d tiles with < %d buildings "
                "(%d remaining).",
                n_dropped, min_building_count, len(tiles_gdf),
            )

    # -- 0b. Feature selection -------------------------------------------
    if auto_exclude:
        clustering_features = get_clustering_features(feature_columns)
        excluded = sorted(set(feature_columns) - set(clustering_features))
        if excluded:
            logger.info(
                "Feature selection: using %d / %d features for clustering. "
                "Excluded: %s",
                len(clustering_features), len(feature_columns),
                ", ".join(excluded),
            )
        else:
            logger.info(
                "Feature selection: all %d features used (none excluded).",
                len(feature_columns),
            )
    else:
        clustering_features = list(feature_columns)
        logger.info(
            "Feature selection: auto_exclude=False, using all %d features.",
            len(feature_columns),
        )

    n_tiles = len(tiles_gdf)
    logger.info(
        "select_patches: %d tiles, %d clustering features.", n_tiles,
        len(clustering_features),
    )

    # -- 1. Standardise features -----------------------------------------
    normed_gdf = normalize_features(tiles_gdf, clustering_features, method="zscore")

    # Extract feature matrix and impute any residual NaN (belt-and-suspenders
    # after normalize_features already handles NaN).
    X = normed_gdf[clustering_features].to_numpy(dtype=float)
    X = _impute_median(X)

    # -- 2. PCA dimensionality reduction ---------------------------------
    pca_scores, pca_loadings, pca_evr = _run_pca(X)

    # -- 3. Determine k --------------------------------------------------
    metadata: dict[str, Any] = {}

    if n_clusters is None:
        sil_info = _find_optimal_k(
            pca_scores, k_range=k_range, random_state=random_state
        )
        n_clusters = sil_info["optimal_k"]
        metadata["silhouette_info"] = sil_info

    # Guard: reduce k if there are fewer tiles than requested clusters.
    if n_clusters >= n_tiles:
        old_k = n_clusters
        n_clusters = max(2, n_tiles - 1)
        logger.warning(
            "Requested k=%d but only %d tiles — reducing to k=%d.",
            old_k,
            n_tiles,
            n_clusters,
        )

    # -- 4. Cluster ------------------------------------------------------
    labels, medoid_indices = _cluster_kmedoids(
        pca_scores, n_clusters=n_clusters, random_state=random_state
    )

    # -- 5. Select representatives ---------------------------------------
    selected_gdf = _select_representatives(
        tiles_gdf, labels, medoid_indices, pca_scores
    )

    # -- 6. Cluster profiles (on original feature space) -----------------
    profiling_gdf = tiles_gdf.copy()
    profiling_gdf["cluster"] = labels.astype(int)
    cluster_profiles = compute_cluster_profiles(
        profiling_gdf, clustering_features, cluster_col="cluster"
    )

    # -- 7. Assemble metadata --------------------------------------------
    metadata.update(
        {
            "n_clusters": n_clusters,
            "clustering_features_used": clustering_features,
            "clustering_features_excluded": sorted(
                set(feature_columns) - set(clustering_features)
            ),
            "pca_explained_variance": pca_evr.tolist(),
            "pca_loadings": pca_loadings.tolist(),
            "cluster_labels": labels.tolist(),
            "pca_scores": pca_scores.tolist(),
            "cluster_profiles": cluster_profiles,
        }
    )

    logger.info(
        "Patch selection complete: %d representatives from %d tiles "
        "(%d clusters).",
        len(selected_gdf),
        n_tiles,
        n_clusters,
    )
    return selected_gdf, metadata

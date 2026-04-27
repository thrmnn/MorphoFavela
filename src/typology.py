"""Settlement typology classification via clustering and threshold flagging."""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_sklearn(module_path: str) -> Any:
    """Lazy-import a scikit-learn submodule, raising a clear error if missing."""
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for typology classification but is not installed. "
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
                # Entire column is NaN — fill with 0
                median_val = 0.0
            col[mask] = median_val
    return arr


def normalize_features(
    gdf: gpd.GeoDataFrame,
    features: list[str],
    method: str = "zscore",
) -> gpd.GeoDataFrame:
    """Normalize feature columns in a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data.
    features : list[str]
        Column names to normalize.
    method : str
        ``"zscore"`` for (x - mean) / std, ``"minmax"`` for (x - min) / (max - min).

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with the feature columns replaced by normalized values.
    """
    gdf = gdf.copy()
    arr = gdf[features].to_numpy(dtype=float)

    # Impute NaN with column median
    arr = _impute_median(arr)

    if method == "zscore":
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        # Guard against zero std (single-value column) — leave as 0
        safe_stds = np.where(stds == 0, 1.0, stds)
        arr = (arr - means) / safe_stds
        # Where std was 0, all values are identical → set to 0
        arr[:, stds == 0] = 0.0
    elif method == "minmax":
        mins = np.min(arr, axis=0)
        maxs = np.max(arr, axis=0)
        ranges = maxs - mins
        safe_ranges = np.where(ranges == 0, 1.0, ranges)
        arr = (arr - mins) / safe_ranges
        # Where range was 0, all values are identical → set to 0
        arr[:, ranges == 0] = 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'zscore' or 'minmax'.")

    gdf[features] = arr
    return gdf


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
) -> dict[str, Any]:
    """Find optimal number of clusters using silhouette analysis.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    k_range : range
        Range of *k* values to evaluate.

    Returns
    -------
    dict
        ``{"k_range": list, "inertia": list, "silhouette": list, "optimal_k": int}``
    """
    cluster_mod = _require_sklearn("sklearn.cluster")
    metrics_mod = _require_sklearn("sklearn.metrics")

    KMeans = cluster_mod.KMeans
    silhouette_score = metrics_mod.silhouette_score

    inertia_list: list[float] = []
    silhouette_list: list[float] = []
    k_list: list[int] = list(k_range)

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        inertia_list.append(float(km.inertia_))
        sil = float(silhouette_score(X, labels))
        silhouette_list.append(sil)
        logger.debug("k=%d  inertia=%.2f  silhouette=%.4f", k, km.inertia_, sil)

    best_idx = int(np.argmax(silhouette_list))
    optimal_k = k_list[best_idx]
    logger.info("Optimal k=%d (silhouette=%.4f)", optimal_k, silhouette_list[best_idx])

    return {
        "k_range": k_list,
        "inertia": inertia_list,
        "silhouette": silhouette_list,
        "optimal_k": optimal_k,
    }


def classify_typology(
    gdf: gpd.GeoDataFrame,
    features: list[str],
    n_clusters: int = 4,
    method: str = "kmeans",
) -> gpd.GeoDataFrame:
    """Cluster zones by morphological features.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data with feature columns.
    features : list[str]
        Column names used for clustering.
    n_clusters : int
        Number of clusters.
    method : str
        ``"kmeans"`` or ``"hierarchical"`` (agglomerative).

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with an added ``cluster`` column (int labels 0 .. n_clusters-1).
    """
    cluster_mod = _require_sklearn("sklearn.cluster")

    gdf = gdf.copy()
    arr = gdf[features].to_numpy(dtype=float)
    arr = _impute_median(arr)

    if method == "kmeans":
        model = cluster_mod.KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(arr)
    elif method == "hierarchical":
        model = cluster_mod.AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(arr)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}. Use 'kmeans' or 'hierarchical'.")

    gdf["cluster"] = labels.astype(int)
    logger.info(
        "Classified %d zones into %d clusters (method=%s)",
        len(gdf),
        n_clusters,
        method,
    )
    return gdf


def flag_zones(
    gdf: gpd.GeoDataFrame,
    thresholds: dict[str, tuple[str, float]] | None = None,
) -> gpd.GeoDataFrame:
    """Flag zones that exceed morphological thresholds.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data.
    thresholds : dict, optional
        Mapping of ``column_name -> (operator, value)`` where operator is
        ``"lt"`` (less than) or ``"gt"`` (greater than).  Defaults to common
        informality indicators.

    Returns
    -------
    GeoDataFrame
        Copy with boolean ``flag_{col}`` columns and an integer ``flag_count``.
    """
    if thresholds is None:
        thresholds = {
            "svf_mean": ("lt", 0.3),
            "bcr": ("gt", 0.6),
            "sigma_h": ("gt", 5.0),
        }

    gdf = gdf.copy()
    flag_cols: list[str] = []

    for col, (op, value) in thresholds.items():
        flag_col = f"flag_{col}"
        if col not in gdf.columns:
            logger.warning(
                "Column %r not found in GeoDataFrame — setting flag_%s to False",
                col,
                col,
            )
            gdf[flag_col] = False
        elif op == "lt":
            gdf[flag_col] = gdf[col] < value
        elif op == "gt":
            gdf[flag_col] = gdf[col] > value
        else:
            raise ValueError(f"Unknown operator {op!r} for column {col!r}. Use 'lt' or 'gt'.")
        flag_cols.append(flag_col)

    gdf["flag_count"] = gdf[flag_cols].sum(axis=1).astype(int)
    return gdf


def compute_cluster_profiles(
    gdf: gpd.GeoDataFrame,
    features: list[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Compute mean metrics per cluster.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain *cluster_col* and all *features* columns.
    features : list[str]
        Feature columns to aggregate.
    cluster_col : str
        Name of the cluster label column.

    Returns
    -------
    DataFrame
        Indexed by cluster, with one column per feature (mean) plus ``count``.
    """
    profiles = gdf.groupby(cluster_col)[features].mean()
    profiles["count"] = gdf.groupby(cluster_col).size()
    return profiles

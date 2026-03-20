"""Spatial statistics for urban morphometric analysis.

Provides reusable functions for spatial autocorrelation and hotspot detection
on GeoDataFrame inputs.  The module wraps PySAL / ESDA primitives into a
consistent, NaN-safe interface:

* **Global Moran's I** — overall spatial autocorrelation (Anselin, 1995).
* **LISA (Local Moran's I)** — local indicators of spatial association
  (Anselin, 1995).
* **Getis-Ord Gi*** — local hotspot / coldspot detection
  (Getis & Ord, 1992).

References
----------
Anselin, L. (1995). Local indicators of spatial association — LISA.
    *Geographical Analysis*, 27(2), 93-115.
Getis, A. & Ord, J. K. (1992). The analysis of spatial association by use
    of distance statistics. *Geographical Analysis*, 24(3), 189-206.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
from esda import Moran, Moran_Local
from esda.getisord import G_Local
from libpysal.weights import DistanceBand, KNN, Queen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spatial weights
# ---------------------------------------------------------------------------


def build_spatial_weights(
    gdf: gpd.GeoDataFrame,
    method: str = "queen",
    k: int = 8,
) -> Any:
    """Construct a row-standardised spatial weight matrix.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geometries.
    method : str
        Weight type — ``"queen"`` (contiguity), ``"knn"`` (k-nearest
        neighbours), or ``"distance"`` (fixed distance band).
    k : int
        Number of neighbours for ``"knn"`` (ignored by other methods).

    Returns
    -------
    libpysal.weights.W
        Row-standardised weight matrix.

    Raises
    ------
    ValueError
        If *method* is not one of the supported values.
    """
    method = method.lower()
    logger.info("Building spatial weights (method=%s, n=%d)", method, len(gdf))

    if method == "queen":
        weights = Queen.from_dataframe(gdf, use_index=False)
    elif method == "knn":
        weights = KNN.from_dataframe(gdf, k=k)
    elif method == "distance":
        # Derive a threshold from the max nearest-neighbour distance
        # (computed on centroids) to guarantee no islands.
        from scipy.spatial import cKDTree

        centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        tree = cKDTree(centroids)
        nn_dists, _ = tree.query(centroids, k=2)  # k=2: self + nearest
        threshold = float(nn_dists[:, 1].max()) * 1.01  # small buffer
        weights = DistanceBand.from_dataframe(gdf, threshold=threshold)
    else:
        raise ValueError(
            f"Unsupported weights method '{method}'. "
            "Choose from 'queen', 'knn', or 'distance'."
        )

    # Handle islands (isolated geometries with no neighbours).
    n_islands = len(weights.islands)
    if n_islands > 0:
        logger.warning(
            "Weight matrix contains %d island(s) with no neighbours.", n_islands
        )

    weights.transform = "r"
    logger.info("Spatial weights built: %d observations.", weights.n)
    return weights


# ---------------------------------------------------------------------------
# Global Moran's I
# ---------------------------------------------------------------------------


def compute_global_morans_i(
    gdf: gpd.GeoDataFrame,
    column: str,
    weights: Any | None = None,
) -> dict[str, float]:
    """Compute Global Moran's I for spatial autocorrelation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geodataframe.
    column : str
        Name of the numeric column to analyse.
    weights : libpysal.weights.W, optional
        Pre-computed weight matrix.  If *None*, queen contiguity weights
        are constructed internally.

    Returns
    -------
    dict
        Keys: ``I``, ``expected_I``, ``p_value``, ``z_score``.
    """
    gdf = gdf.copy()
    valid_mask = gdf[column].notna()
    gdf_valid = gdf.loc[valid_mask].reset_index(drop=True)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN in column '%s'.", n_dropped, column)

    if weights is None:
        weights = build_spatial_weights(gdf_valid)

    y = gdf_valid[column].values
    mi = Moran(y, weights)

    result = {
        "I": float(mi.I),
        "expected_I": float(mi.EI),
        "p_value": float(mi.p_sim),
        "z_score": float(mi.z_sim),
    }
    logger.info(
        "Global Moran's I=%.4f  (p=%.4f, z=%.4f)",
        result["I"],
        result["p_value"],
        result["z_score"],
    )
    return result


# ---------------------------------------------------------------------------
# LISA (Local Moran's I)
# ---------------------------------------------------------------------------


def compute_lisa(
    gdf: gpd.GeoDataFrame,
    column: str,
    weights: Any | None = None,
) -> gpd.GeoDataFrame:
    """Compute Local Moran's I (LISA) clusters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geodataframe.
    column : str
        Name of the numeric column to analyse.
    weights : libpysal.weights.W, optional
        Pre-computed weight matrix.  If *None*, queen contiguity weights
        are constructed internally.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *gdf* with added columns: ``lisa_I``, ``lisa_p``,
        ``lisa_q``, ``lisa_cluster``.
    """
    gdf = gdf.copy()
    valid_mask = gdf[column].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN in column '%s'.", n_dropped, column)

    gdf_valid = gdf.loc[valid_mask].reset_index(drop=True)

    if weights is None:
        weights = build_spatial_weights(gdf_valid)

    y = gdf_valid[column].values
    lisa = Moran_Local(y, weights)

    # Map ESDA quadrant codes to human-readable labels.
    quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

    gdf_valid["lisa_I"] = lisa.Is
    gdf_valid["lisa_p"] = lisa.p_sim
    gdf_valid["lisa_q"] = lisa.q

    clusters = []
    for q, p in zip(lisa.q, lisa.p_sim):
        if p <= 0.05:
            clusters.append(quadrant_labels.get(q, "ns"))
        else:
            clusters.append("ns")
    gdf_valid["lisa_cluster"] = clusters

    # Re-insert NaN rows if any were dropped.
    if n_dropped > 0:
        gdf["lisa_I"] = np.nan
        gdf["lisa_p"] = np.nan
        gdf["lisa_q"] = np.nan
        gdf["lisa_cluster"] = "ns"
        gdf.loc[valid_mask, "lisa_I"] = gdf_valid["lisa_I"].values
        gdf.loc[valid_mask, "lisa_p"] = gdf_valid["lisa_p"].values
        gdf.loc[valid_mask, "lisa_q"] = gdf_valid["lisa_q"].values
        gdf.loc[valid_mask, "lisa_cluster"] = gdf_valid["lisa_cluster"].values
        return gdf

    logger.info(
        "LISA complete: %d significant clusters out of %d.",
        (gdf_valid["lisa_cluster"] != "ns").sum(),
        len(gdf_valid),
    )
    return gdf_valid


# ---------------------------------------------------------------------------
# Getis-Ord Gi*
# ---------------------------------------------------------------------------


def compute_getis_ord(
    gdf: gpd.GeoDataFrame,
    column: str,
    weights: Any | None = None,
) -> gpd.GeoDataFrame:
    """Compute Getis-Ord Gi* hotspot / coldspot classification.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input geodataframe.
    column : str
        Name of the numeric column to analyse.
    weights : libpysal.weights.W, optional
        Pre-computed weight matrix.  If *None*, queen contiguity weights
        are constructed internally.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *gdf* with added columns: ``hotspot_gi`` (z-score),
        ``hotspot_p`` (p-value), ``hotspot_class`` (1 = hotspot,
        -1 = coldspot, 0 = not significant at p <= 0.05).
    """
    gdf = gdf.copy()
    valid_mask = gdf[column].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN in column '%s'.", n_dropped, column)

    gdf_valid = gdf.loc[valid_mask].reset_index(drop=True)

    if weights is None:
        weights = build_spatial_weights(gdf_valid)

    y = gdf_valid[column].values
    gi = G_Local(y, weights)

    gdf_valid["hotspot_gi"] = gi.Zs
    gdf_valid["hotspot_p"] = gi.p_sim

    # Classification.
    classes = np.zeros(len(gdf_valid), dtype=int)
    sig = gdf_valid["hotspot_p"].values <= 0.05
    classes[sig & (gi.Zs > 0)] = 1
    classes[sig & (gi.Zs < 0)] = -1
    gdf_valid["hotspot_class"] = classes

    # Re-insert NaN rows if any were dropped.
    if n_dropped > 0:
        gdf["hotspot_gi"] = np.nan
        gdf["hotspot_p"] = np.nan
        gdf["hotspot_class"] = 0
        gdf.loc[valid_mask, "hotspot_gi"] = gdf_valid["hotspot_gi"].values
        gdf.loc[valid_mask, "hotspot_p"] = gdf_valid["hotspot_p"].values
        gdf.loc[valid_mask, "hotspot_class"] = gdf_valid["hotspot_class"].values
        return gdf

    logger.info(
        "Gi* complete: %d hotspots, %d coldspots out of %d.",
        (gdf_valid["hotspot_class"] == 1).sum(),
        (gdf_valid["hotspot_class"] == -1).sum(),
        len(gdf_valid),
    )
    return gdf_valid

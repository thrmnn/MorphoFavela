"""Crop and export raw source data for each selected CFD patch.

For every selected tile, extracts buildings, roads, DTM, and boundary
geometry into a self-contained directory ready for CFD pre-processing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialisable(obj: Any) -> Any:
    """Convert numpy / shapely types to JSON-serialisable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "__geo_interface__"):
        return mapping(obj)
    return obj


def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a pandas Series to a dict with JSON-safe values."""
    return {k: _serialisable(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

# Columns that are *not* morphometric features (metadata / geometry columns).
_NON_FEATURE_COLS = {
    "tile_id",
    "geometry",
    "geometry_buffered",
    "classification",
    "boundary_overlap_frac",
    "buffer_distance_m",
    "n_buildings",
    "has_dtm_coverage",
    "cluster",
    "selection_reason",
}


def _extract_features(tile_row: pd.Series) -> Dict[str, Any]:
    """Return a dict of morphometric feature values from the tile row."""
    return {
        k: _serialisable(v)
        for k, v in tile_row.items()
        if k not in _NON_FEATURE_COLS and not isinstance(v, (gpd.array.GeometryDtype,))
    }


# ---------------------------------------------------------------------------
# Single-patch export
# ---------------------------------------------------------------------------


def _export_single_patch(
    tile_row: pd.Series,
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    dtm_path: Path,
    patch_dir: Path,
) -> dict:
    """Export cropped source data for a single CFD patch.

    Creates *patch_dir* and writes:
    - ``buildings.gpkg`` -- full building footprints intersecting the buffered
      extent (NOT clipped to boundary).
    - ``dtm.tif`` -- DTM raster cropped to buffered extent (GeoTIFF).
    - ``roads.gpkg`` -- road lines clipped to buffered extent.
    - ``boundary.gpkg`` -- analysis (200x200 m) + buffered polygons.
    - ``metadata.json`` -- tile metadata and feature summary.

    Parameters
    ----------
    tile_row : pd.Series
        Row from the selected_tiles GeoDataFrame.
    buildings_gdf : GeoDataFrame
        Full buildings dataset.
    roads_gdf : GeoDataFrame
        Full roads dataset.
    dtm_path : Path
        Path to the DTM raster (GeoTIFF).
    patch_dir : Path
        Output directory for this patch.

    Returns
    -------
    dict
        Patch metadata (also written to ``metadata.json``).
    """
    _ensure_dir(patch_dir)

    tile_id = tile_row["tile_id"]
    analysis_geom = tile_row["geometry"]
    buffered_geom = tile_row["geometry_buffered"]
    crs = buildings_gdf.crs

    logger.info("Exporting patch %s -> %s", tile_id, patch_dir)

    # --- Buildings: select intersecting, keep full polygons ----------------
    bldg_sindex = buildings_gdf.sindex
    candidates = list(bldg_sindex.intersection(buffered_geom.bounds))
    if candidates:
        nearby = buildings_gdf.iloc[candidates]
        selected_bldg = nearby[nearby.geometry.intersects(buffered_geom)].copy()
    else:
        selected_bldg = buildings_gdf.iloc[0:0].copy()  # empty with schema

    bldg_path = patch_dir / "buildings.gpkg"
    selected_bldg.to_file(bldg_path, driver="GPKG")
    n_buildings = len(selected_bldg)
    logger.info("  buildings.gpkg: %d buildings", n_buildings)

    # --- DTM: crop to buffered extent -------------------------------------
    dtm_tif_path = patch_dir / "dtm.tif"
    dtm_coverage = _export_dtm(dtm_path, buffered_geom, dtm_tif_path)

    # --- Roads: clip to buffered extent -----------------------------------
    roads_path = patch_dir / "roads.gpkg"
    clipped_roads = gpd.clip(roads_gdf, buffered_geom)
    clipped_roads.to_file(roads_path, driver="GPKG")
    logger.info("  roads.gpkg: %d features", len(clipped_roads))

    # --- Boundary: analysis + buffered polygons ---------------------------
    boundary_path = patch_dir / "boundary.gpkg"
    boundary_gdf = gpd.GeoDataFrame(
        {
            "type": ["analysis", "buffered"],
            "geometry": [analysis_geom, buffered_geom],
        },
        crs=crs,
    )
    boundary_gdf.to_file(boundary_path, driver="GPKG")

    # --- Metadata ---------------------------------------------------------
    analysis_bounds = list(analysis_geom.bounds)  # [minx, miny, maxx, maxy]
    buffer_bounds = list(buffered_geom.bounds)

    features = _extract_features(tile_row)

    metadata = {
        "tile_id": str(tile_id),
        "crs": str(crs),
        "analysis_bounds": analysis_bounds,
        "buffer_bounds": buffer_bounds,
        "buffer_distance_m": float(tile_row.get("buffer_distance_m", 0.0)),
        "n_buildings": n_buildings,
        "dtm_coverage": dtm_coverage,
        "selection_reason": _serialisable(tile_row.get("selection_reason", "")),
        "cluster_id": _serialisable(tile_row.get("cluster", None)),
        "classification": str(tile_row.get("classification", "")),
        "features": features,
    }

    meta_path = patch_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=_serialisable)
    logger.info("  metadata.json written")

    return metadata


def _export_dtm(
    dtm_path: Path,
    buffered_geom,
    output_path: Path,
) -> dict:
    """Crop DTM raster to *buffered_geom* and write as GeoTIFF.

    Returns a dict with DTM coverage info (total_pixels, valid_pixels, etc.).
    """
    geo_interface = buffered_geom.__geo_interface__
    dtm_info: Dict[str, Any] = {
        "total_pixels": 0,
        "valid_pixels": 0,
        "coverage_fraction": 0.0,
        "nodata_value": None,
    }

    try:
        with rasterio.open(dtm_path) as src:
            out_image, out_transform = rio_mask(
                src,
                [geo_interface],
                crop=True,
                nodata=src.nodata if src.nodata is not None else np.nan,
            )

            nodata = src.nodata if src.nodata is not None else np.nan
            data = out_image[0]
            total = data.size
            if np.isnan(nodata):
                valid = int(np.isfinite(data).sum())
            else:
                valid = int((data != nodata).sum())

            dtm_info["total_pixels"] = int(total)
            dtm_info["valid_pixels"] = valid
            dtm_info["coverage_fraction"] = round(valid / total, 4) if total > 0 else 0.0
            dtm_info["nodata_value"] = _serialisable(nodata)

            # Write cropped GeoTIFF
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                nodata=nodata,
            )

            with rasterio.open(str(output_path), "w", **profile) as dst:
                dst.write(out_image)

        logger.info(
            "  dtm.tif: %d/%d valid pixels (%.1f%%)",
            valid, total, dtm_info["coverage_fraction"] * 100,
        )

    except Exception as exc:
        logger.warning("  DTM export failed for %s: %s", output_path, exc)
        dtm_info["error"] = str(exc)

    return dtm_info


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    patches_metadata: List[dict],
    output_dir: Path,
) -> Path:
    """Write ``manifest.json`` summarising all exported patches.

    Parameters
    ----------
    patches_metadata : list[dict]
        List of per-patch metadata dicts (returned by ``_export_single_patch``).
    output_dir : Path
        Root export directory.

    Returns
    -------
    Path
        Path to the written ``manifest.json``.
    """
    n_patches = len(patches_metadata)
    n_buildings_total = sum(m.get("n_buildings", 0) for m in patches_metadata)

    coverage_fracs = [
        m.get("dtm_coverage", {}).get("coverage_fraction", 0.0)
        for m in patches_metadata
    ]
    mean_dtm_coverage = float(np.mean(coverage_fracs)) if coverage_fracs else 0.0

    manifest = {
        "n_patches": n_patches,
        "summary": {
            "total_buildings": n_buildings_total,
            "mean_dtm_coverage": round(mean_dtm_coverage, 4),
            "patches_with_full_dtm": sum(1 for f in coverage_fracs if f >= 0.99),
            "patches_with_no_buildings": sum(
                1 for m in patches_metadata if m.get("n_buildings", 0) == 0
            ),
        },
        "patches": patches_metadata,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_serialisable)

    logger.info("Wrote manifest.json: %d patches, %d total buildings.",
                n_patches, n_buildings_total)
    return manifest_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_patches(
    selected_tiles: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    dtm_path: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> Path:
    """Export cropped source data for every selected CFD patch.

    For each tile in *selected_tiles*, creates a sub-directory under
    *output_dir* and writes buildings, roads, DTM, boundary, and metadata
    files.  Finally writes a ``manifest.json`` summarising all exports.

    Parameters
    ----------
    selected_tiles : GeoDataFrame
        Tiles selected for CFD simulation.  Must contain ``tile_id``,
        ``geometry``, ``geometry_buffered``, ``classification``,
        ``cluster``, ``selection_reason``, ``buffer_distance_m``, and
        morphometric feature columns.
    buildings_gdf : GeoDataFrame
        Full buildings dataset (EPSG:31983).
    roads_gdf : GeoDataFrame
        Full roads dataset (EPSG:31983).
    dtm_path : Path
        Path to the DTM raster (GeoTIFF).
    output_dir : Path
        Root directory for patch exports.
    n_jobs : int
        Number of parallel workers.  1 = sequential (default).

    Returns
    -------
    Path
        Path to the written ``manifest.json``.
    """
    output_dir = Path(output_dir)
    dtm_path = Path(dtm_path)
    _ensure_dir(output_dir)

    logger.info(
        "Exporting %d patches to %s (n_jobs=%d).",
        len(selected_tiles), output_dir, n_jobs,
    )

    if n_jobs > 1:
        from joblib import Parallel, delayed

        patches_metadata = Parallel(n_jobs=n_jobs)(
            delayed(_export_single_patch)(
                row,
                buildings_gdf,
                roads_gdf,
                dtm_path,
                output_dir / f"patch_{row['tile_id']}",
            )
            for _, row in selected_tiles.iterrows()
        )
    else:
        patches_metadata = []
        for _, row in selected_tiles.iterrows():
            patch_dir = output_dir / f"patch_{row['tile_id']}"
            meta = _export_single_patch(
                row, buildings_gdf, roads_gdf, dtm_path, patch_dir,
            )
            patches_metadata.append(meta)

    manifest_path = _write_manifest(patches_metadata, output_dir)
    logger.info("Export complete. Manifest: %s", manifest_path)
    return manifest_path

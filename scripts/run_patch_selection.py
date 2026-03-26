#!/usr/bin/env python3
"""
CFD Patch Selection Framework.

Tiles a settlement into analysis zones, computes morphometric features per
tile, clusters tiles by similarity, selects representative patches for CFD
simulation, and exports cropped source data for each selected patch.

Usage:
    python scripts/run_patch_selection.py --area maré --mode all
    python scripts/run_patch_selection.py --area maré --mode tile
    python scripts/run_patch_selection.py --area maré --mode features
    python scripts/run_patch_selection.py --area maré --mode select --n-clusters 8
    python scripts/run_patch_selection.py --area maré --mode export
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_output_dir
from src.svf_v2.paths import resolve_boundary, resolve_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_patch_selection")


def main():
    parser = argparse.ArgumentParser(description="CFD Patch Selection Framework")
    parser.add_argument("--area", required=True, help="Area name (e.g. maré)")
    parser.add_argument("--tile-size", type=float, default=200.0, help="Tile size in metres")
    parser.add_argument("--overlap", type=float, default=0.0, help="Tile overlap in metres")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Number of clusters (default: auto via silhouette)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for export (1=serial, -1=all)")
    parser.add_argument(
        "--mode", default="all",
        choices=["tile", "features", "select", "export", "visualize", "all"],
    )
    parser.add_argument("--skip-visual", action="store_true", help="Skip visualization")
    parser.add_argument("--height-field", default="altura", help="Building height column")
    args = parser.parse_args()

    # Resolve paths
    dtm_path, footprints_path, roads_path = resolve_paths(args.area)
    boundary_path = resolve_boundary(args.area)
    output_dir = get_area_output_dir(args.area) / "patch_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Area: %s", args.area)
    logger.info("Output: %s", output_dir)

    modes = [args.mode] if args.mode != "all" else ["tile", "features", "select", "export", "visualize"]

    # Load data once
    import rasterio
    buildings_gdf = gpd.read_file(footprints_path)
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
    if buildings_gdf.crs != dtm_crs:
        buildings_gdf = buildings_gdf.to_crs(dtm_crs)

    roads_gdf = gpd.read_file(roads_path)
    if roads_gdf.crs != dtm_crs:
        roads_gdf = roads_gdf.to_crs(dtm_crs)

    boundary_gdf = None
    if boundary_path:
        boundary_gdf = gpd.read_file(boundary_path)
        if boundary_gdf.crs != dtm_crs:
            boundary_gdf = boundary_gdf.to_crs(dtm_crs)
    else:
        # Fallback: use building convex hull as boundary
        from shapely.ops import unary_union
        hull = unary_union(buildings_gdf.geometry.values).convex_hull
        boundary_gdf = gpd.GeoDataFrame({"geometry": [hull]}, crs=dtm_crs)
        logger.warning("No boundary file — using building convex hull.")

    tiles_path = output_dir / "tiles.gpkg"
    features_path = output_dir / "tiles_features.gpkg"
    selected_path = output_dir / "selected_patches.gpkg"
    metadata_path = output_dir / "selection_metadata.json"

    # ── TILE ──────────────────────────────────────────────────────────────
    if "tile" in modes:
        from src.patch_selection.tiling import build_tile_grid

        logger.info("=" * 60)
        logger.info("STEP 1: TILING")
        logger.info("=" * 60)
        t0 = time.time()
        tiles = build_tile_grid(
            boundary_gdf, buildings_gdf, dtm_path,
            tile_size=args.tile_size, overlap=args.overlap,
        )
        # Store extra geometry columns as WKT — GPKG only supports one geometry column
        tiles_save = tiles.copy()
        for gcol in ("geometry_buffered", "geometry_unclipped"):
            if gcol in tiles_save.columns:
                tiles_save[f"{gcol}_wkt"] = tiles_save[gcol].apply(
                    lambda g: g.wkt if g is not None else None
                )
                tiles_save = tiles_save.drop(columns=[gcol])
        tiles_save.to_file(tiles_path, driver="GPKG")
        logger.info("Saved %d tiles to %s (%.1fs)", len(tiles), tiles_path.name, time.time() - t0)

    # ── FEATURES ──────────────────────────────────────────────────────────
    if "features" in modes:
        from src.patch_selection.features import compute_tile_features

        logger.info("=" * 60)
        logger.info("STEP 2: FEATURES")
        logger.info("=" * 60)
        if "tiles" not in dir():
            tiles = gpd.read_file(tiles_path)
            # Reconstruct geometry_buffered from WKT if present
            if "geometry_buffered_wkt" in tiles.columns:
                from shapely import wkt
                tiles["geometry_buffered"] = tiles["geometry_buffered_wkt"].apply(
                    lambda s: wkt.loads(s) if s else None
                )

        # Check for existing SVF results
        svf_path = get_area_output_dir(args.area) / "svf_v2" / "svf_streets.gpkg"
        if not svf_path.exists():
            svf_path = None
            logger.info("No SVF results found — SVF features will be NaN.")

        t0 = time.time()
        tiles_feat = compute_tile_features(
            tiles, buildings_gdf, streets=roads_gdf,
            height_col=args.height_field,
            dtm_path=dtm_path, svf_path=svf_path,
        )
        # Rejoin geometry from tiles (features is a plain DataFrame)
        geom_cols = ["tile_id", "geometry"]
        if "geometry_buffered" in tiles.columns:
            geom_cols.append("geometry_buffered")
        tiles_feat_gdf = tiles[geom_cols].merge(
            tiles_feat, on="tile_id", how="left",
        )
        tiles_feat_gdf = gpd.GeoDataFrame(tiles_feat_gdf, geometry="geometry", crs=dtm_crs)

        # Save — geometry_buffered as WKT for GPKG compat
        save_gdf = tiles_feat_gdf.copy()
        if "geometry_buffered" in save_gdf.columns:
            save_gdf["geometry_buffered_wkt"] = save_gdf["geometry_buffered"].apply(
                lambda g: g.wkt if g is not None else None
            )
            save_gdf = save_gdf.drop(columns=["geometry_buffered"])
        save_gdf.to_file(features_path, driver="GPKG")
        tiles_feat_gdf.drop(columns=["geometry", "geometry_buffered"], errors="ignore").to_csv(
            output_dir / "tiles_features.csv", index=False
        )
        logger.info("Saved features to %s (%.1fs)", features_path.name, time.time() - t0)

    # ── SELECT ────────────────────────────────────────────────────────────
    if "select" in modes:
        from src.patch_selection.selection import select_patches

        logger.info("=" * 60)
        logger.info("STEP 3: SELECTION")
        logger.info("=" * 60)
        if "tiles_feat_gdf" not in dir():
            tiles_feat_gdf = gpd.read_file(features_path)

        # Identify numeric feature columns (exclude metadata/geometry)
        skip_cols = {
            "tile_id", "geometry", "geometry_buffered", "classification",
            "boundary_overlap_frac", "n_buildings", "has_dtm_coverage",
            "buffer_distance_m", "cluster", "pc1", "pc2", "selection_reason",
        }
        feature_cols = [
            c for c in tiles_feat_gdf.columns
            if c not in skip_cols
            and tiles_feat_gdf[c].dtype in ("float64", "float32", "int64")
            and tiles_feat_gdf[c].notna().sum() >= 3
        ]
        logger.info("Feature columns for clustering: %s", feature_cols)

        t0 = time.time()
        selected, metadata = select_patches(
            tiles_feat_gdf, feature_cols, n_clusters=args.n_clusters,
        )
        selected.to_file(selected_path, driver="GPKG")
        with open(metadata_path, "w") as f:
            import numpy as np
            def _convert(obj):
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
            json.dump(metadata, f, indent=2, default=_convert)

        logger.info(
            "Selected %d patches (k=%s) → %s (%.1fs)",
            len(selected), metadata.get("optimal_k", "?"), selected_path.name, time.time() - t0,
        )
        for _, row in selected.iterrows():
            logger.info("  %s: cluster=%s reason=%s", row.tile_id, row.get("cluster"), row.selection_reason)

    # ── EXPORT ────────────────────────────────────────────────────────────
    if "export" in modes:
        from src.patch_selection.export import export_patches

        logger.info("=" * 60)
        logger.info("STEP 4: EXPORT")
        logger.info("=" * 60)
        if "selected" not in dir():
            selected = gpd.read_file(selected_path)

        t0 = time.time()
        exports_dir = output_dir / "exports"
        manifest_path = export_patches(
            selected, buildings_gdf, roads_gdf, dtm_path,
            output_dir=exports_dir, n_jobs=args.n_jobs,
        )
        logger.info("Exported patches → %s (%.1fs)", manifest_path, time.time() - t0)

    # ── VISUALIZE ─────────────────────────────────────────────────────────
    if "visualize" in modes and not args.skip_visual:
        logger.info("=" * 60)
        logger.info("STEP 5: VISUALIZATION")
        logger.info("=" * 60)
        # Load results if not in memory
        if "tiles_feat_gdf" not in dir():
            tiles_feat_gdf = gpd.read_file(features_path)
        if "selected" not in dir():
            selected = gpd.read_file(selected_path)
        if "metadata" not in dir() and metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        try:
            from src.patch_selection.visualize import (
                plot_tile_clusters,
                plot_pca_scatter,
                plot_feature_distributions,
                plot_dtm_coverage,
            )

            # Merge cluster labels + PCA scores into all tiles for visualization
            if "selected" in dir() and "cluster" in selected.columns:
                cluster_cols = ["tile_id", "cluster"]
                if "pc1" in selected.columns:
                    cluster_cols.extend(["pc1", "pc2"])
                cluster_map = selected[cluster_cols].drop_duplicates("tile_id")
                # Get cluster for ALL tiles from the selection metadata
                if "metadata" in dir() and "cluster_labels" in metadata:
                    import numpy as _np
                    labels = metadata["cluster_labels"]
                    tiles_feat_gdf["cluster"] = labels[:len(tiles_feat_gdf)]
                if "metadata" in dir() and "pca_scores" in metadata:
                    scores = _np.array(metadata["pca_scores"])
                    if scores.shape[0] == len(tiles_feat_gdf) and scores.shape[1] >= 2:
                        tiles_feat_gdf["pc1"] = scores[:, 0]
                        tiles_feat_gdf["pc2"] = scores[:, 1]

            # Tile cluster map
            plot_tile_clusters(
                tiles_feat_gdf, selected, output_dir / "tile_clusters.png",
                boundary=boundary_gdf, buildings=buildings_gdf,
            )

            # PCA scatter
            ev = metadata.get("pca_explained_variance") if "metadata" in dir() else None
            plot_pca_scatter(
                tiles_feat_gdf, selected, output_dir / "pca_scatter.png",
                explained_variance=ev,
            )

            # Feature distributions per cluster
            skip_cols = {
                "tile_id", "geometry", "geometry_buffered", "geometry_buffered_wkt",
                "classification", "boundary_overlap_frac", "n_buildings",
                "has_dtm_coverage", "buffer_distance_m", "cluster", "pc1", "pc2",
                "selection_reason",
            }
            feat_cols = [
                c for c in tiles_feat_gdf.columns
                if c not in skip_cols and tiles_feat_gdf[c].dtype in ("float64", "float32", "int64")
            ]
            if feat_cols:
                plot_feature_distributions(
                    tiles_feat_gdf, feat_cols, output_dir / "feature_distributions.png",
                )

            # DTM coverage map
            plot_dtm_coverage(
                tiles_feat_gdf, output_dir / "dtm_coverage.png",
                boundary=boundary_gdf,
            )

            logger.info("Saved visualizations to %s", output_dir)
        except Exception as e:
            logger.warning("Visualization failed: %s", e, exc_info=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()

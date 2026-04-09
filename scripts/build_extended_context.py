#!/usr/bin/env python3
"""Build extended building and terrain context for CFD domain coverage.

Clips city-wide Rio building footprints and DTM to a buffer zone around
a favela boundary, merges with site-specific data (preferring site data
where overlap exists), and saves the extended datasets.

This increases the candidate patch pool for CFD sampling by providing
building data beyond the favela perimeter — 250 m CFD domains near
the edge are no longer excluded for lack of context geometry.

Usage:
    python scripts/build_extended_context.py --area vidigal
    python scripts/build_extended_context.py --area vidigal --buffer 400
    python scripts/build_extended_context.py --area rocinha --buffer 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPECTED_CRS, get_area_data_dir  # noqa: E402
from src.svf_v2.paths import resolve_boundary, resolve_paths  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("extended_context")

# City-wide data paths (shared across all sites)
RJ_DATA_DIR = PROJECT_ROOT / "data" / "RJ"
RJ_BUILDINGS = RJ_DATA_DIR / "buildings_RJ_2019.shp"
RJ_DTM = RJ_DATA_DIR / "DTM_RJ.tif"


# ═══════════════════════════════════════════════════════════════════
#  1. Extended buildings
# ═══════════════════════════════════════════════════════════════════


def build_extended_buildings(
    area: str,
    buffer_m: float = 300.0,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Build extended building dataset for a site.

    Parameters
    ----------
    area : str
        Site identifier (e.g., 'vidigal', 'rocinha').
    buffer_m : float
        Buffer distance in metres around the site boundary.

    Returns
    -------
    extended : GeoDataFrame
        Merged building footprints (site + context).
    output_path : Path
        Where the file was saved.
    stats : dict
        Summary statistics for reporting.
    """
    # Load site boundary
    boundary_path = resolve_boundary(area)
    if boundary_path is None:
        raise FileNotFoundError(f"No boundary found for area: {area}")
    boundary = gpd.read_file(boundary_path)
    boundary_geom = boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)

    # Load site buildings
    _, fp_path, _ = resolve_paths(area)
    site_bld = gpd.read_file(fp_path)
    logger.info("Site buildings: %d features.", len(site_bld))

    # Load city-wide buildings — bbox filter for speed (avoids reading 2.4M rows)
    logger.info(
        "Loading city-wide buildings (bbox filter, buffer=%dm)...", buffer_m
    )
    if not RJ_BUILDINGS.exists():
        raise FileNotFoundError(f"City-wide buildings not found: {RJ_BUILDINGS}")
    minx, miny, maxx, maxy = buffer_geom.bounds
    rj_bld = gpd.read_file(RJ_BUILDINGS, bbox=(minx, miny, maxx, maxy))
    logger.info("  Candidates in bbox: %d.", len(rj_bld))

    # Precise clip to buffer polygon
    rj_bld = rj_bld[rj_bld.intersects(buffer_geom)].copy()
    logger.info("  Within buffer polygon: %d.", len(rj_bld))

    # Deduplicate: remove city-wide buildings that overlap site buildings.
    # The Vidigal extract comes from the same RJ dataset, but may have
    # been refined (TLS-corrected heights, manual fixes).  Keep the
    # site version for any overlapping footprint.
    n_overlap = 0
    if not site_bld.empty and not rj_bld.empty:
        overlap_idx = gpd.sjoin(
            rj_bld,
            site_bld[["geometry"]],
            how="inner",
            predicate="intersects",
        ).index.unique()
        n_overlap = len(overlap_idx)
        rj_context = rj_bld.drop(overlap_idx, errors="ignore").copy()
    else:
        rj_context = rj_bld
    logger.info(
        "  Dedup: %d overlapping removed, %d context buildings remain.",
        n_overlap,
        len(rj_context),
    )

    # Harmonize height columns
    for df in [site_bld, rj_context]:
        if "altura" in df.columns and "height" not in df.columns:
            df["height"] = df["altura"]
        if "base" in df.columns and "base_height" not in df.columns:
            df["base_height"] = df["base"]

    # Tag source
    site_bld["_source"] = "site"
    rj_context["_source"] = "context"

    # Merge on common columns
    common_cols = sorted(set(site_bld.columns) & set(rj_context.columns))
    if "geometry" not in common_cols:
        common_cols.append("geometry")

    extended = gpd.GeoDataFrame(
        pd.concat(
            [site_bld[common_cols], rj_context[common_cols]], ignore_index=True
        ),
        crs=site_bld.crs,
    )

    # Repair invalid geometries
    invalid = ~extended.geometry.is_valid
    if invalid.any():
        logger.warning("Repairing %d invalid geometries.", invalid.sum())
        extended.loc[invalid, "geometry"] = extended.loc[
            invalid, "geometry"
        ].buffer(0)

    # Save
    data_dir = get_area_data_dir(area).parent  # data/{area}/
    output_path = data_dir / f"buildings_extended_{int(buffer_m)}m.gpkg"
    extended.to_file(output_path, driver="GPKG")
    logger.info("Saved %s (%d features).", output_path, len(extended))

    # Statistics — use convex hull of centroids for robustness (avoids
    # TopologyException on sites like Rocinha with complex footprints)
    try:
        site_hull_area = site_bld.geometry.union_all().convex_hull.area
    except Exception:
        logger.warning("union_all failed for site buildings — using centroid hull.")
        from shapely.geometry import MultiPoint
        site_hull_area = MultiPoint(
            site_bld.geometry.centroid.tolist()
        ).convex_hull.area

    try:
        ext_hull_area = extended.geometry.union_all().convex_hull.area
    except Exception:
        logger.warning("union_all failed for extended buildings — using centroid hull.")
        from shapely.geometry import MultiPoint
        ext_hull_area = MultiPoint(
            extended.geometry.centroid.tolist()
        ).convex_hull.area
    stats = {
        "site_buildings": len(site_bld),
        "context_buildings": len(rj_context),
        "overlap_removed": n_overlap,
        "total_buildings": len(extended),
        "site_coverage_m2": float(site_hull_area),
        "extended_coverage_m2": float(ext_hull_area),
        "coverage_gain_m2": float(ext_hull_area - site_hull_area),
        "coverage_gain_pct": float(
            (ext_hull_area - site_hull_area) / site_hull_area * 100
        ),
        "buffer_m": buffer_m,
        "output_path": str(output_path),
    }

    logger.info(
        "Coverage: %.0f m² → %.0f m² (+%.1f%%).",
        stats["site_coverage_m2"],
        stats["extended_coverage_m2"],
        stats["coverage_gain_pct"],
    )
    return extended, output_path, stats


# ═══════════════════════════════════════════════════════════════════
#  2. Extended DTM
# ═══════════════════════════════════════════════════════════════════


def build_extended_dtm(
    area: str,
    buffer_m: float = 300.0,
) -> tuple[Path, dict]:
    """Build extended DTM for a site by merging site DTM with city-wide DTM.

    The site-specific raster takes priority where it has data (may be
    ALS-derived and more accurate).

    Returns
    -------
    output_path : Path
        Where the merged raster was saved.
    stats : dict
        Summary statistics.
    """
    # Load boundary + buffer
    boundary_path = resolve_boundary(area)
    boundary = gpd.read_file(boundary_path)
    boundary_geom = boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)
    buffer_bounds = buffer_geom.bounds  # (minx, miny, maxx, maxy)

    # Site DTM
    site_dtm_path, _, _ = resolve_paths(area)

    if not RJ_DTM.exists():
        raise FileNotFoundError(f"City-wide DTM not found: {RJ_DTM}")

    logger.info("Merging DTM: site + city-wide (buffer=%dm)...", buffer_m)

    # rasterio.merge with site first → site takes priority ("first" method)
    src_site = rasterio.open(site_dtm_path)
    src_rj = rasterio.open(RJ_DTM)

    # Determine nodata from sources
    nodata = src_site.nodata if src_site.nodata is not None else -9999.0

    mosaic, out_transform = rio_merge(
        [src_site, src_rj],
        bounds=buffer_bounds,
        res=src_site.res,
        method="first",
        nodata=nodata,
    )

    src_site.close()
    src_rj.close()

    # Sanitise: replace any residual float32-max sentinels with nodata
    sentinel_mask = np.abs(mosaic) > 1e10
    if sentinel_mask.any():
        logger.warning(
            "Replacing %d sentinel pixels with nodata (%.1f).",
            sentinel_mask.sum(),
            nodata,
        )
        mosaic[sentinel_mask] = nodata

    # Save
    data_dir = get_area_data_dir(area).parent
    output_path = data_dir / f"dtm_extended_{int(buffer_m)}m.tif"

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": mosaic.shape[2],
        "height": mosaic.shape[1],
        "count": mosaic.shape[0],
        "crs": EXPECTED_CRS,
        "transform": out_transform,
        "compress": "lzw",
        "nodata": nodata,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic.astype(np.float32))

    logger.info("Saved %s (%d x %d pixels).", output_path, profile["width"], profile["height"])

    stats = {
        "width": profile["width"],
        "height": profile["height"],
        "res_m": src_site.res[0] if hasattr(src_site, "res") else 5.0,
        "output_path": str(output_path),
    }
    return output_path, stats


# ═══════════════════════════════════════════════════════════════════
#  3. Diagnostic figure
# ═══════════════════════════════════════════════════════════════════


def generate_diagnostic_figure(
    area: str,
    extended: gpd.GeoDataFrame,
    buffer_m: float,
    output_dir: Path,
) -> Path:
    """Show favela boundary, original vs context buildings, buffer zone."""
    from src.cartography import (
        add_north_arrow,
        add_scale_bar,
        apply_publication_style,
        format_utm_axes,
    )

    apply_publication_style()

    boundary_path = resolve_boundary(area)
    boundary = gpd.read_file(boundary_path)
    boundary_geom = boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Buffer zone
    gpd.GeoSeries([buffer_geom], crs=extended.crs).boundary.plot(
        ax=ax, color="#1f77b4", linewidth=1.5, linestyle=":", label=f"{int(buffer_m)}m buffer"
    )

    # Context buildings
    ctx = extended[extended["_source"] == "context"]
    if not ctx.empty:
        ctx.plot(
            ax=ax,
            facecolor="#ffcc80",
            edgecolor="#e65100",
            linewidth=0.2,
            alpha=0.7,
            label=f"Context buildings ({len(ctx)})",
            zorder=2,
        )

    # Site buildings
    site = extended[extended["_source"] == "site"]
    if not site.empty:
        site.plot(
            ax=ax,
            facecolor="#90caf9",
            edgecolor="#1565c0",
            linewidth=0.2,
            alpha=0.8,
            label=f"Site buildings ({len(site)})",
            zorder=3,
        )

    # Boundary
    boundary.boundary.plot(
        ax=ax, color="k", linewidth=2, linestyle="--", label="Favela boundary", zorder=4
    )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    area_display = area.replace("_", " ").title()
    ax.set_title(
        f"Extended Building Context \u2014 {area_display} ({int(buffer_m)}m buffer)",
        fontsize=14,
        fontweight="bold",
    )
    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)
    ax.set_aspect("equal")

    fig_path = output_dir / f"extended_context_{int(buffer_m)}m.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostic figure: %s", fig_path)
    return fig_path


# ═══════════════════════════════════════════════════════════════════
#  4. Orchestrator
# ═══════════════════════════════════════════════════════════════════


def build_extended_context(
    area: str,
    buffer_m: float = 300.0,
    skip_figure: bool = False,
) -> dict:
    """Build extended buildings + DTM for a site.

    Returns a dict with paths to all generated files and summary stats.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("BUILDING EXTENDED CONTEXT: %s (%dm buffer)", area.upper(), buffer_m)
    logger.info("=" * 60)

    # Buildings
    extended, bld_path, bld_stats = build_extended_buildings(area, buffer_m)

    # DTM
    dtm_path, dtm_stats = build_extended_dtm(area, buffer_m)

    # Diagnostic figure
    fig_path = None
    if not skip_figure:
        from src.config import get_area_output_dir

        fig_dir = get_area_output_dir(area) / "cfd"
        fig_path = generate_diagnostic_figure(area, extended, buffer_m, fig_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("EXTENDED CONTEXT COMPLETE in %.1f s", elapsed)
    logger.info("  Buildings: %s", bld_path)
    logger.info("  DTM:       %s", dtm_path)
    if fig_path:
        logger.info("  Figure:    %s", fig_path)
    logger.info("=" * 60)

    return {
        "buildings_path": str(bld_path),
        "dtm_path": str(dtm_path),
        "figure_path": str(fig_path) if fig_path else None,
        "building_stats": bld_stats,
        "dtm_stats": dtm_stats,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Build extended building + DTM context for CFD sampling",
    )
    parser.add_argument(
        "--area",
        required=True,
        help="Site name (e.g., vidigal, rocinha)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=300.0,
        help="Buffer distance in metres (default: 300)",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip diagnostic figure generation",
    )
    args = parser.parse_args()

    build_extended_context(args.area, buffer_m=args.buffer, skip_figure=args.skip_figure)


if __name__ == "__main__":
    main()

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
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds as window_from_bounds

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPECTED_CRS, get_area_data_dir, get_area_output_dir  # noqa: E402
from src.morphometry.invariants import phantom_mask  # noqa: E402
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
#  0. Context resolution (registered site, municipal favela, or polygon)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Context:
    """Everything the builders need, resolved once from --area or --polygon.

    ``site_dtm_path`` / ``site_fp_path`` are None when there is no per-site
    directory — the builders then fall back to the municipal RJ layers.
    """

    name: str
    boundary: gpd.GeoDataFrame
    site_dtm_path: Path | None
    site_fp_path: Path | None
    output_dir: Path
    fig_dir: Path


def resolve_context(area: str | None = None, polygon: str | Path | None = None) -> Context:
    if (area is None) == (polygon is None):
        raise ValueError("Provide exactly one of --area or --polygon.")

    if polygon is not None:
        boundary = gpd.read_file(polygon)
        if boundary.crs is not None and boundary.crs.to_string() != EXPECTED_CRS:
            raise ValueError(f"Polygon must be {EXPECTED_CRS}, got {boundary.crs.to_string()}.")
        name = Path(polygon).stem
        output_dir = PROJECT_ROOT / "data" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        return Context(name, boundary, None, None, output_dir, PROJECT_ROOT / "outputs" / name / "cfd")

    boundary_path = resolve_boundary(area)
    if boundary_path is None:
        raise FileNotFoundError(f"No boundary found for area: {area}")
    boundary = gpd.read_file(boundary_path)

    try:
        dtm_path, fp_path, _ = resolve_paths(area)
        output_dir = get_area_data_dir(area).parent
        fig_dir = get_area_output_dir(area) / "cfd"
        return Context(area, boundary, dtm_path, fp_path, output_dir, fig_dir)
    except ValueError:
        # Unregistered favela resolved via the municipal limits layer:
        # no per-site DTM/footprints, so the builders read from data/RJ/.
        from src.svf_v2.paths import _slugify

        slug = _slugify(area)
        output_dir = PROJECT_ROOT / "data" / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        return Context(area, boundary, None, None, output_dir, PROJECT_ROOT / "outputs" / slug / "cfd")


# ═══════════════════════════════════════════════════════════════════
#  1. Extended buildings
# ═══════════════════════════════════════════════════════════════════


def build_extended_buildings(
    ctx: Context,
    buffer_m: float = 300.0,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Build extended building dataset for a site.

    Parameters
    ----------
    ctx : Context
        Resolved boundary + per-site paths (or None → municipal fallback).
    buffer_m : float
        Buffer distance in metres around the site boundary.

    Returns
    -------
    extended : GeoDataFrame
        Merged building footprints (site + context), or municipal-only when
        there is no per-site footprint file.
    output_path : Path
        Where the file was saved.
    stats : dict
        Summary statistics for reporting.
    """
    boundary_geom = ctx.boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)

    # Load site buildings (absent when falling back to municipal data)
    if ctx.site_fp_path is not None:
        site_bld = gpd.read_file(ctx.site_fp_path)
        logger.info("Site buildings: %d features.", len(site_bld))
    else:
        site_bld = None
        logger.info("No per-site footprints — using municipal buildings only.")

    # Load city-wide buildings — bbox filter for speed (avoids reading 2.4M rows)
    logger.info("Loading city-wide buildings (bbox filter, buffer=%dm)...", buffer_m)
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
    if site_bld is not None and not site_bld.empty and not rj_bld.empty:
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
    frames = [rj_context] if site_bld is None else [site_bld, rj_context]
    for df in frames:
        if "altura" in df.columns and "height" not in df.columns:
            df["height"] = df["altura"]
        if "base" in df.columns and "base_height" not in df.columns:
            df["base_height"] = df["base"]

    rj_context["_source"] = "context"

    if site_bld is None:
        # Municipal-only: keep all city columns (altura, tipo, topo, …)
        extended = rj_context.copy()
    else:
        site_bld["_source"] = "site"
        common_cols = sorted(set(site_bld.columns) & set(rj_context.columns))
        if "geometry" not in common_cols:
            common_cols.append("geometry")
        extended = gpd.GeoDataFrame(
            pd.concat([site_bld[common_cols], rj_context[common_cols]], ignore_index=True),
            crs=rj_context.crs,
        )

    # Repair invalid geometries
    invalid = ~extended.geometry.is_valid
    if invalid.any():
        logger.warning("Repairing %d invalid geometries.", invalid.sum())
        extended.loc[invalid, "geometry"] = extended.loc[invalid, "geometry"].buffer(0)

    # Drop degenerate geometries — buffer(0) above can collapse a degenerate
    # polygon to empty, and null/empty footprints violate the data contract
    # (Polygon/MultiPolygon required) and silently corrupt λp / coverage stats.
    degenerate = extended.geometry.is_empty | extended.geometry.isna()
    if degenerate.any():
        logger.warning("Dropping %d empty/null geometries before write.", degenerate.sum())
        extended = extended[~degenerate].copy()

    # Drop registry-corrupt heights — rows with topo (absolute top elevation)
    # == 0 carry an altura mis-derived as the base elevation, which extrudes
    # into phantom towers up to 232 m. The SVF scene builder already filters
    # these (src/svf_v2/scene.py); shared detector in src.morphometry.invariants.
    # ``extruding_only`` keeps zero-height topo==0 rows, matching the historical
    # ``(topo==0) & (altura>0)`` predicate (behaviour-preserving on the grids).
    phantom = phantom_mask(extended, extruding_only=True)
    if phantom.any():
        logger.warning(
            "Dropping %d topo==0 phantom-height building(s) before write.",
            int(phantom.sum()),
        )
        extended = extended[~phantom].copy()

    # Save
    output_path = ctx.output_dir / f"buildings_extended_{int(buffer_m)}m.gpkg"
    extended.to_file(output_path, driver="GPKG")
    logger.info("Saved %s (%d features).", output_path, len(extended))

    # Statistics — use convex hull of centroids for robustness (avoids
    # TopologyException on sites like Rocinha with complex footprints)
    if site_bld is None or site_bld.empty:
        site_hull_area = 0.0
    else:
        try:
            site_hull_area = site_bld.geometry.union_all().convex_hull.area
        except Exception:
            logger.warning("union_all failed for site buildings — using centroid hull.")
            from shapely.geometry import MultiPoint

            site_hull_area = MultiPoint(site_bld.geometry.centroid.tolist()).convex_hull.area

    try:
        ext_hull_area = extended.geometry.union_all().convex_hull.area
    except Exception:
        logger.warning("union_all failed for extended buildings — using centroid hull.")
        from shapely.geometry import MultiPoint

        ext_hull_area = MultiPoint(extended.geometry.centroid.tolist()).convex_hull.area
    gain_pct = (
        float((ext_hull_area - site_hull_area) / site_hull_area * 100)
        if site_hull_area > 0
        else float("nan")
    )
    stats = {
        "site_buildings": 0 if site_bld is None else len(site_bld),
        "context_buildings": len(rj_context),
        "overlap_removed": n_overlap,
        "total_buildings": len(extended),
        "site_coverage_m2": float(site_hull_area),
        "extended_coverage_m2": float(ext_hull_area),
        "coverage_gain_m2": float(ext_hull_area - site_hull_area),
        "coverage_gain_pct": gain_pct,
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
    ctx: Context,
    buffer_m: float = 300.0,
) -> tuple[Path, dict]:
    """Build extended DTM for a site.

    With a per-site DTM, merges it over the 5 m city-wide raster
    (site takes priority where it has data — it may be ALS-derived and
    more accurate). Without one, window-clips the 5 m ``DTM_RJ.tif``
    directly to the buffered bounds (no resampling — the 5 m grid is kept
    honest).

    Returns
    -------
    output_path : Path
        Where the raster was saved.
    stats : dict
        Summary statistics.
    """
    boundary_geom = ctx.boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)
    buffer_bounds = buffer_geom.bounds  # (minx, miny, maxx, maxy)

    if not RJ_DTM.exists():
        raise FileNotFoundError(f"City-wide DTM not found: {RJ_DTM}")

    if ctx.site_dtm_path is not None:
        logger.info("Merging DTM: site + city-wide (buffer=%dm)...", buffer_m)

        # rasterio.merge with site first → site takes priority ("first" method)
        src_site = rasterio.open(ctx.site_dtm_path)
        src_rj = rasterio.open(RJ_DTM)

        nodata = src_site.nodata if src_site.nodata is not None else -9999.0
        res_m = src_site.res[0]

        mosaic, out_transform = rio_merge(
            [src_site, src_rj],
            bounds=buffer_bounds,
            res=src_site.res,
            method="first",
            nodata=nodata,
        )

        src_site.close()
        src_rj.close()
    else:
        logger.info("No per-site DTM — window-clipping 5 m DTM_RJ (buffer=%dm)...", buffer_m)
        with rasterio.open(RJ_DTM) as src_rj:
            window = window_from_bounds(*buffer_bounds, transform=src_rj.transform)
            window = window.round_offsets().round_lengths()
            mosaic = src_rj.read(window=window)
            out_transform = src_rj.window_transform(window)
            res_m = src_rj.res[0]
        nodata = -9999.0  # DTM_RJ fills with a float32-max sentinel; normalise it

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
    output_path = ctx.output_dir / f"dtm_extended_{int(buffer_m)}m.tif"

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
        "res_m": res_m,
        "output_path": str(output_path),
    }
    return output_path, stats


# ═══════════════════════════════════════════════════════════════════
#  3. Diagnostic figure
# ═══════════════════════════════════════════════════════════════════


def generate_diagnostic_figure(
    ctx: Context,
    extended: gpd.GeoDataFrame,
    buffer_m: float,
) -> Path:
    """Show favela boundary, original vs context buildings, buffer zone."""
    from src.cartography import (
        add_north_arrow,
        add_scale_bar,
        apply_publication_style,
        format_utm_axes,
    )

    apply_publication_style()

    boundary = ctx.boundary
    boundary_geom = boundary.geometry.union_all()
    buffer_geom = boundary_geom.buffer(buffer_m)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Buffer zone
    gpd.GeoSeries([buffer_geom], crs=extended.crs).boundary.plot(
        ax=ax,
        color="#1f77b4",
        linewidth=1.5,
        linestyle=":",
        label=f"{int(buffer_m)}m buffer",
    )

    # Context buildings
    ctx_bld = extended[extended["_source"] == "context"]
    if not ctx_bld.empty:
        ctx_bld.plot(
            ax=ax,
            facecolor="#ffcc80",
            edgecolor="#e65100",
            linewidth=0.2,
            alpha=0.7,
            label=f"Context buildings ({len(ctx_bld)})",
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
    area_display = ctx.name.replace("_", " ").title()
    ax.set_title(
        f"Extended Building Context \u2014 {area_display} ({int(buffer_m)}m buffer)",
        fontsize=14,
        fontweight="bold",
    )
    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)
    ax.set_aspect("equal")

    fig_path = ctx.fig_dir / f"extended_context_{int(buffer_m)}m.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostic figure: %s", fig_path)
    return fig_path


# ═══════════════════════════════════════════════════════════════════
#  4. Orchestrator
# ═══════════════════════════════════════════════════════════════════


def build_extended_context(
    area: str | None = None,
    buffer_m: float = 300.0,
    skip_figure: bool = False,
    polygon: str | Path | None = None,
) -> dict:
    """Build extended buildings + DTM for a site or an arbitrary polygon.

    Pass either ``area`` (registered site or municipal favela name/code) or
    ``polygon`` (path to a one-feature EPSG:31983 vector). When no per-site
    directory exists, inputs fall back to the municipal ``data/RJ/`` layers.

    Returns a dict with paths to all generated files and summary stats.
    """
    ctx = resolve_context(area=area, polygon=polygon)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("BUILDING EXTENDED CONTEXT: %s (%dm buffer)", ctx.name.upper(), buffer_m)
    logger.info("=" * 60)

    # Buildings
    extended, bld_path, bld_stats = build_extended_buildings(ctx, buffer_m)

    # DTM
    dtm_path, dtm_stats = build_extended_dtm(ctx, buffer_m)

    # Diagnostic figure
    fig_path = None
    if not skip_figure:
        fig_path = generate_diagnostic_figure(ctx, extended, buffer_m)

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
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--area",
        help="Registered site (e.g., vidigal) or municipal favela name/cod_favela",
    )
    src.add_argument(
        "--polygon",
        help="Path to a one-feature EPSG:31983 boundary; bypasses the registry",
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

    build_extended_context(
        area=args.area,
        polygon=args.polygon,
        buffer_m=args.buffer,
        skip_figure=args.skip_figure,
    )


if __name__ == "__main__":
    main()

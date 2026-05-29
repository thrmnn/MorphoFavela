#!/usr/bin/env python3
"""Stratified CFD pilot batch sampling from a 10 m morphometric grid.

Selects 12-15 CFD simulation patch locations using stratified sampling
across SVF x slope x lambda_p feature space.  Designed for reuse across
multiple study sites -- only the CONFIG block changes per site.

Usage:
    python scripts/run_pilot_sampling.py                  # default: vidigal
    python scripts/run_pilot_sampling.py --site rocinha    # other site preset
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.spatial.distance import cdist
from shapely.geometry import Point, box, mapping

# Analysis patch geometry: 100 m-diameter circle centred on the patch point.
# Circles (not squares) give isotropic morphometric averaging and let the
# same patch be re-used across all 8 wind-direction meshes (see §6).
PATCH_RADIUS_M = 50.0
PATCH_DIAMETER_M = 2.0 * PATCH_RADIUS_M  # = W_patch in the v1 manifest
PATCH_AREA_M2 = math.pi * PATCH_RADIUS_M**2  # ≈ 7853.98 m²

# Reference radius for the CFD domain circle in debug plots. Matches the
# canonical `cfd_domain_radius` (patch_meta); the production domain is the
# rectangular v1 extents below — this circle is a visual reference only.
CFD_DOMAIN_RADIUS_M = 250.0

# V1 rectangular per-direction CFD domain (Franke / COST 732 /
# Tominaga 2008 AIJ / Blocken 2015 wide-obstacle scheme).
# Canonical contract: src/cfd_integration/rectangular_domain_v1.json.
V1_BLOCKAGE_GATE = 0.05  # AIJ Tominaga 2008
V1_LATERAL_FLOOR_FACTOR = 5.0  # lateral ≥ 5 · W_patch
V1_SAFETY_MARGIN_M = 50.0  # additive on the rotated-rect diagonal


def _v1_domain_extents(h_max: float) -> dict:
    """Per-patch v1 rectangular domain, in metres. See manifest for derivation."""
    R = PATCH_RADIUS_M
    W = PATCH_DIAMETER_M
    upstream = 5.0 * h_max + R
    downstream = 15.0 * h_max + R
    lateral = max(5.0 * h_max + R, V1_LATERAL_FLOOR_FACTOR * W)
    top = 5.0 * h_max
    cross_section = 2.0 * lateral * top
    frontal = PATCH_DIAMETER_M * h_max
    blockage_ratio = frontal / cross_section if cross_section > 0 else float("inf")
    half_long = 10.0 * h_max + R
    half_short = lateral
    source_data_required = math.ceil(math.sqrt(half_long**2 + half_short**2)) + V1_SAFETY_MARGIN_M
    return {
        "domain_upstream_m": upstream,
        "domain_downstream_m": downstream,
        "domain_lateral_m": lateral,
        "domain_top_m": top,
        "domain_blockage_frontal_m2": frontal,
        "domain_blockage_cross_section_m2": cross_section,
        "domain_blockage_ratio": blockage_ratio,
        "domain_blockage_ok": blockage_ratio < V1_BLOCKAGE_GATE,
        "source_data_required_m": source_data_required,
    }


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cartography import (  # noqa: E402
    add_north_arrow,
    add_scale_bar,
    add_terrain_contours,
    apply_publication_style,
    format_utm_axes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pilot_sampling")


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — only this block changes per site
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # --- Site identity ---
    "site_id": "vidigal",
    "site_prefix": "VDG",
    # --- Input paths (relative to PROJECT_ROOT) ---
    "grid_metrics": "outputs/vidigal/morphometrics/grid/grid_metrics.gpkg",
    "buildings": "data/vidigal/raw/Vidigal_buildings.shp",
    "boundary": "data/vidigal/raw/Vidigal_Limit.shp",
    "dtm": "data/vidigal/raw/DTM_Vidigal.tif",
    # --- Output root (script creates subdirectories) ---
    "output_dir": "outputs/vidigal/sampling_cfd/pilot_sampling",
    # --- Stratification bins ---
    "svf_bins": [0.0, 0.15, 0.30, 1.0],
    "svf_labels": ["SVF<0.15", "0.15≤SVF<0.30", "SVF≥0.30"],
    "slope_bins": [0.0, 15.0, 90.0],
    "slope_labels": ["slope<15°", "slope≥15°"],
    "lambda_p_bins": [0.0, 0.5, 1.0],
    "lambda_p_labels": ["λp<0.5", "λp≥0.5"],
    # --- Domain geometry ---
    "analysis_patch_size": 100,  # metres, diameter of circular analysis zone
    # V1 rectangular per-direction. The eligibility filter no longer uses a
    # cylindrical 250 m domain proxy; instead it checks that the source-data
    # convex hull covers a fixed worst-case envelope around each candidate
    # patch (the rectangular envelope is per-patch and not knowable until
    # H_max is computed, so we use a fixed conservative radius here that
    # bounds the campaign maximum of 646 m).
    "data_coverage_radius": 700,  # metres — matches buildings_extended_700m.gpkg
    # --- Selection constraints ---
    "min_patch_spacing": 80,  # metres between patch centres
    "min_building_coverage": 0.50,  # fraction of analysis patch
    "min_domain_data_coverage": 0.70,  # fraction of data_coverage_radius circle
    # --- Allocation ---
    "target_patches": [12, 15],  # [min, max] total patches
    "min_per_stratum": 1,  # minimum per occupied stratum
    "bonus_strata_count": 3,  # top N most populated get +1
    # --- Column name mapping (adapt per site if schemas differ) ---
    "col_svf": "svf",
    "col_lambda_p": "lambda_p",
    "col_slope": "slope_deg",
    "col_porosity": "porosity",
    "col_sigma_h": "sigma_h",
    "col_h_mean": "H_mean",
    "col_entropy": "street_orientation_entropy",
    "col_far": "far",
    "col_building_count": "building_count",
    # --- CRS ---
    "crs": "EPSG:31983",
    # --- Reproducibility ---
    "random_seed": 42,
    # --- Figures ---
    "dpi": 300,
}

# Site presets — select via  --site <name>  CLI argument.
# Each preset overrides only the site-specific keys in CONFIG.
SITE_PRESETS: dict[str, dict] = {
    "vidigal": {},  # CONFIG default
    "rocinha": {
        "site_id": "rocinha",
        "site_prefix": "ROC",
        "grid_metrics": "outputs/rocinha/morphometrics/grid/grid_metrics.gpkg",
        "buildings": "data/rocinha/raw/rocinha_buildings.shp",
        "boundary": "data/rocinha/raw/rocinha_boundary.shp",
        "dtm": "data/rocinha/raw/rocinha_dtm.tif",
        "output_dir": "outputs/rocinha/sampling_cfd/pilot_sampling",
    },
    "riodaspedras": {
        "site_id": "riodaspedras",
        "site_prefix": "RDP",
        "grid_metrics": "outputs/riodaspedras/morphometrics/grid/grid_metrics.gpkg",
        "buildings": "data/riodaspedras/raw/riodaspedras_buildings.shp",
        "boundary": "data/riodaspedras/raw/riodaspedras_boundary.shp",
        "dtm": "data/riodaspedras/raw/riodaspedras_dtm.tif",
        "output_dir": "outputs/riodaspedras/sampling_cfd/pilot_sampling",
    },
    "complexo_do_alemao": {
        "site_id": "complexo_do_alemao",
        "site_prefix": "CDA",
        "grid_metrics": "outputs/complexo_do_alemao/morphometrics/grid/grid_metrics.gpkg",
        "buildings": "data/complexo_do_alemao/raw/complexo_do_alemao_buildings.shp",
        "boundary": "data/complexo_do_alemao/raw/complexo_do_alemao_boundary.shp",
        "dtm": "data/complexo_do_alemao/raw/complexo_do_alemao_dtm.tif",
        "output_dir": "outputs/complexo_do_alemao/sampling_cfd/pilot_sampling",
    },
    "maré": {
        "site_id": "maré",
        "site_prefix": "MAR",
        "grid_metrics": "outputs/maré/morphometrics/grid/grid_metrics.gpkg",
        "buildings": "data/maré/raw/buildings_mare.shp",
        "boundary": "data/maré/raw/mare_boundary.shp",
        "dtm": "data/maré/raw/mare_dtm.tif",
        "output_dir": "outputs/maré/sampling_cfd/pilot_sampling",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _resolve(path_str: str) -> Path:
    """Resolve a relative path against PROJECT_ROOT."""
    return PROJECT_ROOT / path_str


def _md5(path: Path) -> str:
    """Return the MD5 hex-digest of *path*."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str | None:
    """Return the current HEAD commit hash, or *None*."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _hillshade(
    dtm: np.ndarray, res: float, azimuth: float = 315, altitude: float = 45
) -> np.ndarray:
    """Compute an analytical hillshade from a DTM array."""
    dy, dx = np.gradient(dtm, res)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad, alt_rad = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(
        az_rad - aspect
    )
    return np.clip(shade, 0, 1)


def _safe_bins(bins: list[float]) -> list[float]:
    """Extend the last bin edge by epsilon so pd.cut(right=False) includes it."""
    out = list(bins)
    out[-1] += 1e-6
    return out


# ═══════════════════════════════════════════════════════════════════
#  1.  Load and validate
# ═══════════════════════════════════════════════════════════════════


def load_and_validate(
    config: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    """Load grid metrics, building footprints, and (optional) boundary.

    Validates that expected columns exist in the grid and that CRS is
    consistent across all layers.
    """
    target_epsg = int(config["crs"].split(":")[1])

    # Grid metrics
    grid_path = _resolve(config["grid_metrics"])
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid metrics not found: {grid_path}")
    grid = gpd.read_file(grid_path)
    logger.info("Loaded grid: %d cells, %d columns.", len(grid), len(grid.columns))

    # Validate required columns
    required = [
        config["col_svf"],
        config["col_lambda_p"],
        config["col_slope"],
        config["col_porosity"],
        config["col_sigma_h"],
        config["col_h_mean"],
        config["col_entropy"],
    ]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise ValueError(
            f"Missing columns in grid: {missing}. Available: {sorted(grid.columns.tolist())}"
        )

    # Ensure centroid columns
    if "centroid_x" not in grid.columns:
        centroids = grid.geometry.centroid
        grid["centroid_x"] = centroids.x
        grid["centroid_y"] = centroids.y

    # Buildings
    bld_path = _resolve(config["buildings"])
    if not bld_path.exists():
        raise FileNotFoundError(f"Buildings not found: {bld_path}")
    buildings = gpd.read_file(bld_path)
    invalid = ~buildings.geometry.is_valid
    if invalid.any():
        logger.warning("Repairing %d invalid building geometries.", invalid.sum())
        buildings.loc[invalid, "geometry"] = buildings.loc[invalid, "geometry"].buffer(0)
    logger.info("Loaded buildings: %d features.", len(buildings))

    # CRS alignment
    for label, gdf in [("grid", grid), ("buildings", buildings)]:
        if gdf.crs is None or gdf.crs.to_epsg() != target_epsg:
            logger.warning("Reprojecting %s to EPSG:%d.", label, target_epsg)
            gdf = gdf.to_crs(epsg=target_epsg)
            if label == "grid":
                grid = gdf
            else:
                buildings = gdf

    # Boundary (optional)
    boundary = None
    bnd_path = _resolve(config["boundary"])
    if bnd_path.exists():
        boundary = gpd.read_file(bnd_path)
        if boundary.crs is None or boundary.crs.to_epsg() != target_epsg:
            boundary = boundary.to_crs(epsg=target_epsg)
        logger.info("Loaded boundary: %d features.", len(boundary))

    return grid, buildings, boundary


# ═══════════════════════════════════════════════════════════════════
#  2.  Stratification
# ═══════════════════════════════════════════════════════════════════


def assign_strata(grid: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:
    """Assign each cell to one of the 12 strata (SVF x slope x lambda_p)."""
    grid = grid.copy()

    svf = grid[config["col_svf"]]
    slope = grid[config["col_slope"]]
    lp = grid[config["col_lambda_p"]]

    n_svf = len(config["svf_labels"])
    n_slope = len(config["slope_labels"])
    n_lp = len(config["lambda_p_labels"])

    # Numeric bin indices (1-based)
    svf_idx = pd.cut(
        svf,
        bins=_safe_bins(config["svf_bins"]),
        labels=range(1, n_svf + 1),
        include_lowest=True,
        right=False,
    )
    slope_idx = pd.cut(
        slope,
        bins=_safe_bins(config["slope_bins"]),
        labels=range(1, n_slope + 1),
        include_lowest=True,
        right=False,
    )
    lp_idx = pd.cut(
        lp,
        bins=_safe_bins(config["lambda_p_bins"]),
        labels=range(1, n_lp + 1),
        include_lowest=True,
        right=False,
    )

    # Descriptive bin labels
    grid["svf_bin"] = pd.cut(
        svf,
        bins=_safe_bins(config["svf_bins"]),
        labels=config["svf_labels"],
        include_lowest=True,
        right=False,
    )
    grid["slope_bin"] = pd.cut(
        slope,
        bins=_safe_bins(config["slope_bins"]),
        labels=config["slope_labels"],
        include_lowest=True,
        right=False,
    )
    grid["lambda_p_bin"] = pd.cut(
        lp,
        bins=_safe_bins(config["lambda_p_bins"]),
        labels=config["lambda_p_labels"],
        include_lowest=True,
        right=False,
    )

    nan_mask = svf_idx.isna() | slope_idx.isna() | lp_idx.isna()

    grid["stratum_id"] = pd.NA
    ok = ~nan_mask
    grid.loc[ok, "stratum_id"] = (
        "SVF"
        + svf_idx[ok].astype(str)
        + "_SLP"
        + slope_idx[ok].astype(str)
        + "_LP"
        + lp_idx[ok].astype(str)
    )

    if nan_mask.any():
        logger.warning(
            "%d cells have NaN in stratification variables — excluded.",
            nan_mask.sum(),
        )

    n_strata = grid["stratum_id"].dropna().nunique()
    logger.info(
        "Assigned %d possible strata; %d are populated.",
        n_svf * n_slope * n_lp,
        n_strata,
    )
    return grid


# ═══════════════════════════════════════════════════════════════════
#  3.  Edge exclusion
# ═══════════════════════════════════════════════════════════════════


def exclude_edges(
    grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Remove cells where domain or analysis patch has insufficient coverage.

    Two-pass filter:
      1. Domain data coverage — a worst-case-radius circle (config
         ``data_coverage_radius``, default 700 m to bound the v1
         rectangular source-data envelope at ~646 m) must overlap
         ≥70 % of the building-data convex hull.
      2. Building coverage — 100 m-diameter circular analysis patch must
         have ≥50 % footprint coverage.
    """
    radius = config["data_coverage_radius"]
    min_bld_cov = config["min_building_coverage"]
    min_dom_cov = config["min_domain_data_coverage"]

    # Building data extent — convex hull of all footprints
    logger.info("Computing building data extent (convex hull)...")
    data_extent = buildings.geometry.union_all().convex_hull

    # Pre-filter: inner = data_extent eroded by radius
    inner = data_extent.buffer(-radius)
    inner_valid = inner.is_valid and not inner.is_empty

    # --- Pass 1: domain data coverage ---
    logger.info("Checking domain data coverage (%d m radius)...", radius)
    n = len(grid)
    domain_coverage = np.full(n, np.nan)

    for i in range(n):
        cx, cy = grid.iloc[i]["centroid_x"], grid.iloc[i]["centroid_y"]
        pt = Point(cx, cy)
        if inner_valid and inner.contains(pt):
            domain_coverage[i] = 1.0
        else:
            circle = pt.buffer(radius)
            inter = circle.intersection(data_extent)
            domain_coverage[i] = inter.area / circle.area

    grid = grid.copy()
    grid["_domain_coverage"] = domain_coverage

    n_excl_dom = int((domain_coverage < min_dom_cov).sum())
    logger.info(
        "  Domain coverage exclusion: %d / %d cells (%.1f%%).",
        n_excl_dom,
        n,
        100 * n_excl_dom / n,
    )

    eligible = grid[grid["_domain_coverage"] >= min_dom_cov].copy()

    # --- Pass 2: building coverage in analysis patches ---
    logger.info(
        "Computing building coverage in %.0f m-diameter circular patches (%d candidates)...",
        2 * PATCH_RADIUS_M,
        len(eligible),
    )
    bld_sindex = buildings.sindex
    bld_cov = np.zeros(len(eligible))

    for idx, (_, row) in enumerate(eligible.iterrows()):
        cx, cy = row["centroid_x"], row["centroid_y"]
        patch = Point(cx, cy).buffer(PATCH_RADIUS_M)
        candidates = list(bld_sindex.intersection(patch.bounds))
        if not candidates:
            continue
        total = 0.0
        for bidx in candidates:
            bgeom = buildings.geometry.iloc[bidx]
            if patch.intersects(bgeom):
                total += patch.intersection(bgeom).area
        bld_cov[idx] = total / PATCH_AREA_M2

    eligible["_building_coverage"] = bld_cov
    n_excl_bld = int((bld_cov < min_bld_cov).sum())
    logger.info("  Building coverage exclusion: %d cells.", n_excl_bld)

    # Also exclude NaN strata
    n_excl_nan = int(eligible["stratum_id"].isna().sum())

    eligible = eligible[
        (eligible["_building_coverage"] >= min_bld_cov) & eligible["stratum_id"].notna()
    ].copy()

    stats = {
        "edge_domain": n_excl_dom,
        "low_building_coverage": n_excl_bld,
        "nan_strata": n_excl_nan,
        "total_excluded": len(grid) - len(eligible),
        "total_eligible": len(eligible),
    }
    logger.info(
        "Eligible cells: %d / %d (%.1f%%).",
        len(eligible),
        len(grid),
        100 * len(eligible) / max(len(grid), 1),
    )
    return eligible, stats


# ═══════════════════════════════════════════════════════════════════
#  4.  Stratum summary and allocation
# ═══════════════════════════════════════════════════════════════════


def build_strata_summary(
    grid_full: gpd.GeoDataFrame,
    grid_eligible: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build the stratum population summary table."""
    rows = []
    for i, svf_l in enumerate(config["svf_labels"], 1):
        for j, slp_l in enumerate(config["slope_labels"], 1):
            for k, lp_l in enumerate(config["lambda_p_labels"], 1):
                rows.append(
                    {
                        "stratum_id": f"SVF{i}_SLP{j}_LP{k}",
                        "SVF_bin": svf_l,
                        "slope_bin": slp_l,
                        "lambda_p_bin": lp_l,
                    }
                )

    summary = pd.DataFrame(rows)

    total_counts = grid_full.groupby("stratum_id").size()
    elig_counts = grid_eligible.groupby("stratum_id").size()

    summary["n_cells_total"] = summary["stratum_id"].map(total_counts).fillna(0).astype(int)
    summary["n_cells_eligible"] = summary["stratum_id"].map(elig_counts).fillna(0).astype(int)

    total = summary["n_cells_total"].sum()
    summary["pct_total"] = (100 * summary["n_cells_total"] / total).round(1) if total > 0 else 0.0
    summary["is_empty"] = summary["n_cells_eligible"] == 0
    return summary


def allocate_patches(summary: pd.DataFrame, config: dict) -> dict[str, int]:
    """Decide how many patches each stratum receives."""
    target_min, target_max = config["target_patches"]
    min_per = config["min_per_stratum"]
    bonus_n = config["bonus_strata_count"]

    allocation: dict[str, int] = {}
    for _, row in summary.iterrows():
        sid = row["stratum_id"]
        allocation[sid] = min_per if row["n_cells_eligible"] > 0 else 0

    total = sum(allocation.values())

    # Bonus to most-populated strata
    occupied = summary[summary["n_cells_eligible"] > 0].sort_values(
        "n_cells_eligible", ascending=False
    )
    bonus_given = 0
    for _, row in occupied.iterrows():
        if total >= target_max or bonus_given >= bonus_n:
            break
        allocation[row["stratum_id"]] += 1
        total += 1
        bonus_given += 1

    # If still below minimum, keep adding to most-populated
    if total < target_min:
        for _, row in occupied.iterrows():
            if total >= target_min:
                break
            allocation[row["stratum_id"]] += 1
            total += 1

    logger.info(
        "Allocated %d patches across %d strata.",
        total,
        sum(1 for v in allocation.values() if v > 0),
    )
    return allocation


# ═══════════════════════════════════════════════════════════════════
#  5.  Within-stratum selection
# ═══════════════════════════════════════════════════════════════════


def select_within_stratum(
    candidates: gpd.GeoDataFrame,
    n: int,
    config: dict,
) -> gpd.GeoDataFrame:
    """Pick *n* patches from a stratum's eligible candidates.

    - n == 1 → cell closest to stratum feature-space centroid (median).
    - n >= 2 → greedy maximin geographic distance.
    """
    if n == 0 or candidates.empty:
        return candidates.iloc[0:0]
    if len(candidates) <= n:
        return candidates

    if n == 1:
        cols = [config["col_svf"], config["col_lambda_p"], config["col_slope"]]
        feats = candidates[cols].values.astype(float)
        std = feats.std(axis=0)
        std[std == 0] = 1.0
        feats_z = (feats - feats.mean(axis=0)) / std
        centroid = np.median(feats_z, axis=0)
        dists = np.linalg.norm(feats_z - centroid, axis=1)
        return candidates.iloc[[int(dists.argmin())]]

    # Maximin geographic distance
    coords = candidates[["centroid_x", "centroid_y"]].values
    dm = cdist(coords, coords)

    # Seed: two most distant cells
    i, j = np.unravel_index(dm.argmax(), dm.shape)
    selected = [int(i), int(j)]

    while len(selected) < n:
        unsel = [k for k in range(len(coords)) if k not in selected]
        min_d = dm[np.ix_(unsel, selected)].min(axis=1)
        best = unsel[int(min_d.argmax())]
        selected.append(best)

    return candidates.iloc[selected]


def enforce_spacing(
    selected: gpd.GeoDataFrame,
    grid_eligible: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    """Drop patches that violate the minimum-spacing constraint.

    Among a conflicting pair, the patch farther from its stratum
    feature-space centroid is dropped.
    """
    min_dist = config["min_patch_spacing"]
    if len(selected) < 2:
        return selected

    coords = selected[["centroid_x", "centroid_y"]].values
    dm = cdist(coords, coords)
    np.fill_diagonal(dm, np.inf)

    # Find all conflicts
    conflicts: list[tuple[int, int, float]] = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if dm[i, j] < min_dist:
                conflicts.append((i, j, dm[i, j]))

    if not conflicts:
        logger.info("All patches pass minimum spacing (>=%d m).", min_dist)
        return selected

    logger.warning("%d spacing conflicts (<%d m) found.", len(conflicts), min_dist)

    # For each cell, compute feature-space distance to its stratum centroid
    cols = [config["col_svf"], config["col_lambda_p"], config["col_slope"]]

    def _feat_dist(idx: int) -> float:
        row = selected.iloc[idx]
        sid = row["stratum_id"]
        stratum_cells = grid_eligible[grid_eligible["stratum_id"] == sid]
        vals = stratum_cells[cols].values.astype(float)
        std = vals.std(axis=0)
        std[std == 0] = 1.0
        centroid = np.median(vals, axis=0)
        cell_vals = np.array([row[c] for c in cols], dtype=float)
        return float(np.linalg.norm((cell_vals - centroid) / std))

    to_drop: set[int] = set()
    for i, j, d in conflicts:
        if i in to_drop or j in to_drop:
            continue
        di, dj = _feat_dist(i), _feat_dist(j)
        to_drop.add(i if di > dj else j)

    keep = [i for i in range(len(selected)) if i not in to_drop]
    result = selected.iloc[keep].copy()
    logger.info("Dropped %d patches for spacing; %d remain.", len(to_drop), len(result))
    return result


# ═══════════════════════════════════════════════════════════════════
#  6.  v1 rectangular-domain extents per patch
# ═══════════════════════════════════════════════════════════════════


def compute_v1_domain(
    selected: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    """Compute per-patch v1 rectangular domain extents + blockage gate.

    Replaces the v0 ``validate_blocken`` cylindrical fetch check. Per-row
    output: H_max_analysis, all four ``domain_*_m`` extents, the blockage
    frontal/cross-section/ratio/ok triple, and ``source_data_required_m``
    (the rotated-rectangle envelope radius). Gate is AIJ Tominaga 2008
    (B < 0.05); see src/cfd_integration/rectangular_domain_v1.json for
    derivations.
    """
    selected = selected.copy()
    bld_sindex = buildings.sindex

    height_col = None
    for col in ["height", "altura"]:
        if col in buildings.columns:
            height_col = col
            break

    rows = []
    for _, row in selected.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        patch = Point(cx, cy).buffer(PATCH_RADIUS_M)
        candidates = list(bld_sindex.intersection(patch.bounds))
        h_max = 0.0
        if candidates and height_col:
            for bidx in candidates:
                bldg = buildings.iloc[bidx]
                if patch.intersects(bldg.geometry):
                    h = bldg.get(height_col, 0)
                    if pd.notna(h) and float(h) > h_max:
                        h_max = float(h)
        rec = {"H_max_analysis": h_max, **_v1_domain_extents(h_max)}
        rows.append(rec)

    domain = pd.DataFrame(rows, index=selected.index)
    for col in domain.columns:
        selected[col] = domain[col].values

    n_fail = int((~selected["domain_blockage_ok"]).sum())
    if n_fail:
        logger.warning(
            "%d patches FAIL v1 blockage gate (B < %.2f):",
            n_fail,
            V1_BLOCKAGE_GATE,
        )
        for _, row in selected[~selected["domain_blockage_ok"]].iterrows():
            logger.warning(
                "  %s: H_max=%.1f m  B=%.3f  lateral=%.0f m",
                row.get("patch_id", "?"),
                row["H_max_analysis"],
                row["domain_blockage_ratio"],
                row["domain_lateral_m"],
            )
    else:
        logger.info(
            "All %d patches pass v1 blockage gate (B < %.2f, silhouette envelope).",
            len(selected),
            V1_BLOCKAGE_GATE,
        )
    return selected


# ═══════════════════════════════════════════════════════════════════
#  7.  Per-patch data export
# ═══════════════════════════════════════════════════════════════════


def export_patch_data(
    selected: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> None:
    """Write buildings, terrain clip, and metadata JSON for each patch.

    Per-patch buildings + DTM clips use ``source_data_required_m`` (the
    rotated worst-case rectangular envelope in v1) so the package is
    self-contained for the per-direction blockMesh. Metadata schema
    matches src/cfd_integration/rectangular_domain_v1.json.
    """
    dtm_path = _resolve(config["dtm"])
    patches_dir = _resolve(config["output_dir"]) / "patches"
    bld_sindex = buildings.sindex

    for _, row in selected.iterrows():
        pid = row["patch_id"]
        cx, cy = row["centroid_x"], row["centroid_y"]
        # Per-patch envelope (covers all 8 wind-direction meshes).
        clip_radius = float(row["source_data_required_m"])
        circle = Point(cx, cy).buffer(clip_radius)

        pdir = patches_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)

        # Buildings within the per-patch envelope
        cands = list(bld_sindex.intersection(circle.bounds))
        domain_blds = buildings.iloc[cands].copy()
        domain_blds = domain_blds[domain_blds.intersects(circle)]
        domain_blds.to_file(pdir / "buildings.gpkg", driver="GPKG")

        # DTM clipped to envelope bounding box
        if dtm_path.exists():
            try:
                with rasterio.open(dtm_path) as src:
                    clipped, transform = rio_mask(
                        src,
                        [mapping(circle)],
                        crop=True,
                        nodata=src.nodata if src.nodata is not None else -9999,
                    )
                    profile = src.profile.copy()
                    profile.update(
                        height=clipped.shape[1],
                        width=clipped.shape[2],
                        transform=transform,
                        tiled=False,
                    )
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)
                    with rasterio.open(pdir / "terrain.tif", "w", **profile) as dst:
                        dst.write(clipped)
            except Exception as e:
                logger.warning("DTM clip failed for %s: %s", pid, e)

        # Metadata JSON — v1 schema per src/cfd_integration/rectangular_domain_v1.json.
        meta = {
            "patch_id": pid,
            "center_x": float(cx),
            "center_y": float(cy),
            "stratum_id": str(row["stratum_id"]),
            "analysis_patch_shape": "circle",
            "analysis_patch_diameter": config["analysis_patch_size"],
            "domain_topology": "rectangular_per_direction",
            "domain_manifest": "src/cfd_integration/rectangular_domain_v1.json",
            config["col_svf"]: _safe_float(row, config["col_svf"]),
            config["col_lambda_p"]: _safe_float(row, config["col_lambda_p"]),
            config["col_slope"]: _safe_float(row, config["col_slope"]),
            config["col_porosity"]: _safe_float(row, config["col_porosity"]),
            config["col_sigma_h"]: _safe_float(row, config["col_sigma_h"]),
            config["col_h_mean"]: _safe_float(row, config["col_h_mean"]),
            "H_max_analysis": float(row["H_max_analysis"]),
            "building_coverage": float(row["_building_coverage"]),
            "domain_data_coverage": float(row["_domain_coverage"]),
            "domain_upstream_m": float(row["domain_upstream_m"]),
            "domain_downstream_m": float(row["domain_downstream_m"]),
            "domain_lateral_m": float(row["domain_lateral_m"]),
            "domain_top_m": float(row["domain_top_m"]),
            "domain_blockage_frontal_m2": float(row["domain_blockage_frontal_m2"]),
            "domain_blockage_cross_section_m2": float(row["domain_blockage_cross_section_m2"]),
            "domain_blockage_ratio": float(row["domain_blockage_ratio"]),
            "domain_blockage_ok": bool(row["domain_blockage_ok"]),
            "source_data_required_m": float(row["source_data_required_m"]),
            "n_buildings_in_envelope": len(domain_blds),
        }
        with open(pdir / "patch_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    logger.info("Exported %d patch packages to %s/", len(selected), patches_dir)


def _safe_float(row, col: str):
    """Extract a float value from a row, returning None for NaN."""
    v = row[col]
    if pd.isna(v):
        return None
    return float(v)


# ═══════════════════════════════════════════════════════════════════
#  8.  Figures
# ═══════════════════════════════════════════════════════════════════

# 12-class categorical palette (colourblind-considerate)
_STRATUM_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
]


def _stratum_cmap(strata_ids: list[str]) -> dict[str, str]:
    unique = sorted(set(s for s in strata_ids if pd.notna(s)))
    return {sid: _STRATUM_COLORS[i % len(_STRATUM_COLORS)] for i, sid in enumerate(unique)}


# --- Figure A: spatial map ---


def generate_figure_map(
    grid: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> Path:
    """Pilot patch locations on stratum-coloured grid with hillshade."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Hillshade background
    dtm_path = _resolve(config["dtm"])
    if dtm_path.exists():
        try:
            with rasterio.open(dtm_path) as src:
                dtm = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    dtm[dtm == nodata] = np.nan
                bounds = src.bounds
                shade = _hillshade(dtm, src.res[0])
                shade = np.where(np.isnan(dtm), 1.0, shade)
                ax.imshow(
                    shade,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    alpha=0.4,
                    zorder=0,
                )
        except Exception as e:
            logger.warning("Hillshade failed: %s", e)

    # Grid cells by stratum
    cmap = _stratum_cmap(grid["stratum_id"].tolist())
    g = grid[grid["stratum_id"].notna()].copy()
    g["_color"] = g["stratum_id"].map(cmap)
    g.plot(ax=ax, color=g["_color"], alpha=0.5, edgecolor="none", zorder=1)

    # Terrain contours
    if dtm_path.exists():
        add_terrain_contours(ax, dtm_path, alpha=0.2)

    # Analysis patches (red dashed circles, 100 m diameter)
    for _, row in selected.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                PATCH_RADIUS_M,
                linewidth=1.5,
                edgecolor="#d62728",
                facecolor="none",
                linestyle="--",
                zorder=5,
            )
        )

    # v1 source-data envelope circles (per-patch radius, blue dotted)
    for _, row in selected.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        r = float(row.get("source_data_required_m", 0.0))
        if r <= 0:
            continue
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                r,
                linewidth=1.0,
                edgecolor="#1f77b4",
                facecolor="none",
                linestyle=":",
                zorder=4,
            )
        )

    # Patch centre markers + labels
    for _, row in selected.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        ax.plot(cx, cy, "ko", markersize=6, zorder=7)
        ax.annotate(
            row["patch_id"],
            (cx, cy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            fontweight="bold",
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )

    # Legend
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=cmap[sid], label=sid, alpha=0.7) for sid in sorted(cmap)]
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle="--",
                lw=1.5,
                label=f"{int(2 * PATCH_RADIUS_M)}m analysis patch (⌀)",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#1f77b4",
                linestyle=":",
                lw=1.0,
                label=f"{r}m CFD domain",
            ),
        ]
    )
    ax.legend(handles=handles, loc="upper left", fontsize=5.5, framealpha=0.9, ncol=2)

    site_name = config["site_id"].replace("_", " ").title()
    ax.set_title(
        f"Pilot CFD Patch Locations \u2014 {site_name} (n={len(selected)})",
        fontsize=14,
        fontweight="bold",
    )

    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)
    ax.set_aspect("equal")

    out = _resolve(config["output_dir"]) / "figures" / "fig_pilot_patches_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure A: %s", out)
    return out


# --- Figure B: feature-space coverage ---


def generate_figure_featurespace(
    grid: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    config: dict,
) -> Path:
    """2x2 scatter grid showing patch coverage of the feature space."""
    apply_publication_style()

    c_svf = config["col_svf"]
    c_lp = config["col_lambda_p"]
    c_slp = config["col_slope"]
    c_sh = config["col_sigma_h"]

    pairs = [
        (c_svf, c_slp, "SVF", "Slope (\u00b0)"),
        (c_svf, c_lp, "SVF", "\u03bbp"),
        (c_slp, c_lp, "Slope (\u00b0)", "\u03bbp"),
        (c_svf, c_sh, "SVF", "\u03c3H (m)"),
    ]

    cmap = _stratum_cmap(selected["stratum_id"].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (xc, yc, xl, yl) in zip(axes.flat, pairs):
        valid = grid[[xc, yc]].dropna()
        ax.scatter(
            valid[xc],
            valid[yc],
            s=3,
            c="#cccccc",
            alpha=0.15,
            zorder=1,
            rasterized=True,
        )
        for _, row in selected.iterrows():
            xv, yv = row[xc], row[yc]
            if pd.isna(xv) or pd.isna(yv):
                continue
            c = cmap.get(row["stratum_id"], "#333333")
            ax.scatter(xv, yv, s=80, c=c, edgecolors="k", linewidths=0.5, zorder=5)
            label = row["patch_id"].split("-")[-1]
            ax.annotate(
                label,
                (xv, yv),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6,
                zorder=6,
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.2)

    site_name = config["site_id"].replace("_", " ").title()
    fig.suptitle(
        f"Feature-Space Coverage \u2014 {site_name}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    out = _resolve(config["output_dir"]) / "figures" / "fig_pilot_patches_featurespace.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure B: %s", out)
    return out


# --- Figure C: per-patch context panels ---


def generate_figure_context(
    selected: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> Path:
    """Grid of building-footprint panels centred on each selected patch."""
    apply_publication_style()

    n = len(selected)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flat

    # Per-patch envelope radius is variable in v1; use a generous fixed
    # view window for visual consistency across patches.
    view_half = max(300, int(config.get("data_coverage_radius", 700)))
    bld_sindex = buildings.sindex

    for i, (_, row) in enumerate(selected.iterrows()):
        ax = axes[i]
        cx, cy = row["centroid_x"], row["centroid_y"]
        vbox = box(cx - view_half, cy - view_half, cx + view_half, cy + view_half)

        cands = list(bld_sindex.intersection(vbox.bounds))
        if cands:
            nearby = buildings.iloc[cands]
            nearby = nearby[nearby.intersects(vbox)]
            nearby.plot(
                ax=ax,
                facecolor="#d0d0d0",
                edgecolor="#666666",
                linewidth=0.3,
                zorder=1,
            )

        # Analysis patch (circle, 100 m diameter)
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                PATCH_RADIUS_M,
                linewidth=1.5,
                edgecolor="#d62728",
                facecolor="none",
                linestyle="--",
                zorder=5,
            )
        )
        # Domain circle
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                CFD_DOMAIN_RADIUS_M,
                linewidth=1.0,
                edgecolor="#1f77b4",
                facecolor="none",
                linestyle=":",
                zorder=4,
            )
        )

        svf = row[config["col_svf"]]
        lp = row[config["col_lambda_p"]]
        slope = row[config["col_slope"]]
        ax.set_title(
            f"{row['patch_id']} | SVF={svf:.2f}  \u03bbp={lp:.2f}  slope={slope:.0f}\u00b0"
            if pd.notna(svf) and pd.notna(lp)
            else row["patch_id"],
            fontsize=7,
            fontweight="bold",
        )

        ax.set_xlim(cx - view_half, cx + view_half)
        ax.set_ylim(cy - view_half, cy + view_half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for j in range(i + 1, nrows * ncols):
        axes[j].set_visible(False)

    site_name = config["site_id"].replace("_", " ").title()
    fig.suptitle(
        f"Patch Context \u2014 {site_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    out = _resolve(config["output_dir"]) / "figures" / "fig_pilot_patches_context.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=config["dpi"], bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure C: %s", out)
    return out


def generate_figures(
    grid: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
) -> dict[str, Path]:
    """Generate all three publication figures."""
    paths: dict[str, Path] = {}
    for name, fn in [
        ("map", generate_figure_map),
        ("featurespace", generate_figure_featurespace),
        ("context", generate_figure_context),
    ]:
        try:
            if name == "context":
                paths[name] = fn(selected, buildings, config)
            elif name == "featurespace":
                paths[name] = fn(grid, selected, config)
            else:
                paths[name] = fn(grid, selected, buildings, config)
        except Exception as e:
            logger.error("Figure '%s' failed: %s", name, e)
    return paths


# ═══════════════════════════════════════════════════════════════════
#  9.  Main orchestrator
# ═══════════════════════════════════════════════════════════════════


def run_sampling(config: dict) -> gpd.GeoDataFrame:
    """Execute the complete pilot sampling pipeline."""
    t0 = time.time()
    np.random.seed(config["random_seed"])

    site = config["site_id"]
    prefix = config["site_prefix"]
    output_dir = _resolve(config["output_dir"])

    logger.info("=" * 60)
    logger.info("CFD PILOT SAMPLING: %s", site.upper())
    logger.info("=" * 60)

    # 1. Load
    grid, buildings, boundary = load_and_validate(config)

    # 2. Stratify
    grid = assign_strata(grid, config)

    # 3. Edge exclusion
    eligible, exclusion_stats = exclude_edges(grid, buildings, config)

    # 4. Summary + allocation
    summary = build_strata_summary(grid, eligible, config)
    allocation = allocate_patches(summary, config)
    summary["n_allocated"] = summary["stratum_id"].map(allocation).fillna(0).astype(int)

    # Print stratum table
    logger.info("")
    logger.info("=" * 85)
    logger.info("STRATUM SUMMARY")
    logger.info(
        "%-20s  %6s  %6s  %5s  %5s  %s",
        "Stratum",
        "Total",
        "Elig.",
        "%Tot",
        "Alloc",
        "Bins",
    )
    logger.info("-" * 85)
    for _, r in summary.iterrows():
        bins_str = f"{r['SVF_bin']} | {r['slope_bin']} | {r['lambda_p_bin']}"
        logger.info(
            "%-20s  %6d  %6d  %5.1f  %5d  %s%s",
            r["stratum_id"],
            r["n_cells_total"],
            r["n_cells_eligible"],
            r["pct_total"],
            r["n_allocated"],
            bins_str,
            "  (EMPTY)" if r["is_empty"] else "",
        )
    logger.info("=" * 85)

    # 5. Select within each stratum (with integrated spacing)
    #    Process strata from most-allocated to least so that the richest
    #    strata get first pick of locations.
    min_spacing = config["min_patch_spacing"]
    strata_order = sorted(allocation, key=lambda s: allocation.get(s, 0), reverse=True)
    all_selected: list[gpd.GeoDataFrame] = []
    selected_coords = np.empty((0, 2))

    for sid in strata_order:
        n = allocation.get(sid, 0)
        if n == 0:
            continue

        candidates = eligible[eligible["stratum_id"] == sid].copy()

        # Remove candidates too close to already-selected patches
        if len(selected_coords) > 0 and len(candidates) > 0:
            cand_xy = candidates[["centroid_x", "centroid_y"]].values
            dists = cdist(cand_xy, selected_coords)
            too_close = dists.min(axis=1) < min_spacing
            candidates = candidates[~too_close]

        if candidates.empty:
            logger.warning("  %s: no candidates remain after spacing filter.", sid)
            continue

        picks = select_within_stratum(candidates, min(n, len(candidates)), config)
        all_selected.append(picks)

        new_xy = picks[["centroid_x", "centroid_y"]].values
        selected_coords = (
            np.vstack([selected_coords, new_xy]) if len(selected_coords) > 0 else new_xy
        )
        logger.info(
            "  %s: selected %d / %d eligible (after spacing).",
            sid,
            len(picks),
            len(eligible[eligible["stratum_id"] == sid]),
        )

    if not all_selected:
        logger.error("No patches selected — check data and parameters.")
        return gpd.GeoDataFrame()

    selected = gpd.GeoDataFrame(pd.concat(all_selected, ignore_index=True), crs=grid.crs)

    # 7. Assign patch IDs
    selected = selected.sort_values(["stratum_id", "centroid_x"]).reset_index(drop=True)
    selected["patch_id"] = [f"{prefix}-P{i + 1:02d}" for i in range(len(selected))]

    # 8. v1 rectangular-domain extents per patch
    selected = compute_v1_domain(selected, buildings, config)

    # Print selected patches
    logger.info("")
    logger.info("=" * 100)
    logger.info("SELECTED PATCHES (%d)", len(selected))
    logger.info(
        "%-8s  %10s  %10s  %5s  %5s  %6s  %5s  %6s  %s",
        "ID",
        "X",
        "Y",
        "SVF",
        "\u03bbp",
        "Slope",
        "Hmax",
        "src.R",
        "Stratum",
    )
    logger.info("-" * 100)
    for _, row in selected.iterrows():
        svf = row[config["col_svf"]]
        lp = row[config["col_lambda_p"]]
        slope = row[config["col_slope"]]
        logger.info(
            "%-8s  %10.1f  %10.1f  %5.2f  %5.2f  %6.1f  %5.1f  %6.0f  %s",
            row["patch_id"],
            row["centroid_x"],
            row["centroid_y"],
            svf if pd.notna(svf) else -1,
            lp if pd.notna(lp) else -1,
            slope if pd.notna(slope) else -1,
            row["H_max_analysis"],
            row["source_data_required_m"],
            row["stratum_id"],
        )
    logger.info("=" * 100)

    # 9. Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "patches").mkdir(exist_ok=True)

    # -- GeoPackage (two layers) --
    patches_gdf = selected.copy()
    patches_gdf["geometry"] = [
        Point(row["centroid_x"], row["centroid_y"]).buffer(PATCH_RADIUS_M)
        for _, row in selected.iterrows()
    ]

    out_cols = [
        "patch_id",
        "centroid_x",
        "centroid_y",
        "stratum_id",
        config["col_svf"],
        config["col_lambda_p"],
        config["col_slope"],
        config["col_porosity"],
        config["col_sigma_h"],
        config["col_h_mean"],
        config["col_entropy"],
        "H_max_analysis",
        "_building_coverage",
        "domain_upstream_m",
        "domain_downstream_m",
        "domain_lateral_m",
        "domain_top_m",
        "domain_blockage_ratio",
        "domain_blockage_ok",
        "source_data_required_m",
        "geometry",
    ]
    out_cols = [c for c in out_cols if c in patches_gdf.columns]
    patches_out = patches_gdf[out_cols].rename(
        columns={
            "centroid_x": "center_x",
            "centroid_y": "center_y",
            "_building_coverage": "building_coverage",
        }
    )

    # v1 envelopes layer: per-patch source-data envelope circle (replaces
    # the v0 cylindrical "cfd_domains" layer).
    envelopes_out = gpd.GeoDataFrame(
        {
            "patch_id": selected["patch_id"].values,
            "source_data_required_m": selected["source_data_required_m"].values,
            "domain_data_coverage": selected["_domain_coverage"].values,
        },
        geometry=[
            Point(row["centroid_x"], row["centroid_y"]).buffer(float(row["source_data_required_m"]))
            for _, row in selected.iterrows()
        ],
        crs=grid.crs,
    )

    gpkg_path = output_dir / "pilot_patches.gpkg"
    patches_out.to_file(gpkg_path, layer="analysis_patches", driver="GPKG")
    envelopes_out.to_file(gpkg_path, layer="source_data_envelopes", driver="GPKG")
    logger.info("Saved %s", gpkg_path)

    # -- CSV --
    csv_df = patches_out.drop(columns="geometry")
    csv_df.to_csv(output_dir / "pilot_patches.csv", index=False)

    # -- Stratum summary CSV --
    summary.to_csv(output_dir / "stratum_summary.csv", index=False)

    # -- Sampling log --
    input_files = {
        "grid_metrics": config["grid_metrics"],
        "buildings": config["buildings"],
        "boundary": config["boundary"],
        "dtm": config["dtm"],
    }
    input_hashes = {}
    for key, path_str in input_files.items():
        p = _resolve(path_str)
        input_hashes[key] = _md5(p) if p.exists() else None

    warnings_list: list[str] = []
    for _, row in selected[~selected["domain_blockage_ok"]].iterrows():
        warnings_list.append(
            f"{row['patch_id']}: v1 blockage gate violated "
            f"(B={row['domain_blockage_ratio']:.3f} ≥ {V1_BLOCKAGE_GATE:.2f})"
        )

    log = {
        "config": {k: v for k, v in config.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script_version": "1.0.0",
        "git_hash": _git_hash(),
        "input_file_hashes": input_hashes,
        "strata_summary": summary.to_dict(orient="records"),
        "excluded_cells": exclusion_stats,
        "selected_patches": [
            {
                "patch_id": row["patch_id"],
                "center_x": float(row["centroid_x"]),
                "center_y": float(row["centroid_y"]),
                "stratum_id": str(row["stratum_id"]),
            }
            for _, row in selected.iterrows()
        ],
        "warnings": warnings_list,
    }
    with open(output_dir / "sampling_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    # 10. Per-patch exports
    export_patch_data(selected, buildings, config)

    # 11. Figures
    generate_figures(grid, selected, buildings, config)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("PILOT SAMPLING COMPLETE in %.1f s", elapsed)
    logger.info("  Patches selected: %d", len(selected))
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 60)

    return selected


# ═══════════════════════════════════════════════════════════════════
#  Multi-site campaign pooling (stub)
# ═══════════════════════════════════════════════════════════════════
#
# The full campaign pools per-site selections and applies cross-site
# maximin spacing.  Interface sketch:
#
#   sites = ["vidigal", "rocinha", "maré", "riodaspedras", "complexo_do_alemao"]
#
#   all_patches = pd.concat([
#       gpd.read_file(
#           f"outputs/{s}/sampling_cfd/pilot_sampling/pilot_patches.gpkg",
#           layer="analysis_patches",
#       )
#       for s in sites
#   ])
#
#   # Within each stratum, apply cross-site maximin spacing:
#   for stratum_id, group in all_patches.groupby("stratum_id"):
#       if len(group) > max_per_stratum:
#           keep = select_within_stratum(group, max_per_stratum, SHARED_CONFIG)
#           # replace group rows with 'keep'
#
#   all_patches.to_file("outputs/cfd_campaign/all_pilot_patches.gpkg")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stratified CFD pilot batch sampling from morphometric grid",
    )
    parser.add_argument(
        "--site",
        default=None,
        choices=list(SITE_PRESETS.keys()),
        help="Site preset (default: vidigal)",
    )
    parser.add_argument(
        "--buildings",
        default=None,
        help="Override building footprints path (e.g., extended context file)",
    )
    parser.add_argument(
        "--dtm",
        default=None,
        help="Override DTM raster path",
    )
    parser.add_argument(
        "--grid",
        default=None,
        help="Override grid metrics path (e.g., from extended morphometric run)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for output directory (e.g., 'extended' → pilot_sampling_extended/)",
    )
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.site and args.site in SITE_PRESETS:
        config.update(SITE_PRESETS[args.site])

    # Apply CLI overrides
    if args.buildings:
        config["buildings"] = args.buildings
    if args.dtm:
        config["dtm"] = args.dtm
    if args.grid:
        config["grid_metrics"] = args.grid
    if args.output_suffix:
        base = config["output_dir"]
        config["output_dir"] = base.rstrip("/") + "_" + args.output_suffix

    run_sampling(config)


if __name__ == "__main__":
    main()

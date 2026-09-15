"""WP-03: TLS validity floors for the C' ground claim (G2 alley-width, G1 lite facade).

Spec: docs/wp03_tls_spec.md. No .rcp registration report is on disk for the
Vidigal TLS scans, so the registration floor that would normally come from a
vendor RMSE is instead a MEASURED residual: the TLS ground raster (built here
from the three e57 scans via PDAL) against the ALS DTM that
wp02_surface.build_surface already treats as ground truth for C'. G2 and
G1-lite both run wp02_horizon.patch_visibility with the SAME sky
discretization (P1_SKY_PATCHES) on the ALS 2.5D surface (wp04_sites'
build_site_surface, unmodified) and the TLS DSM built here.

Run: python -m src.brisa_solar.wp03_tls [--data-root PATH] [--run-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
from affine import Affine
from rasterio.features import rasterize
from rasterio.merge import merge as rio_merge
from scipy import ndimage
from scipy.signal import fftconvolve
from shapely import STRtree
from shapely.geometry import Point

from .constants import P1_SKY_PATCHES, REPO_ROOT
from .wp02_horizon import default_device, hemisphere_mask, patch_visibility, svf_solid_angle, write_run_manifest
from .wp04_sites import CELL_M, MAX_DIST_M, OBS_HEIGHT_M, build_site_surface, resolve_native_paths
from src.svf_v2 import sampling as svf_sampling
from src.svf_v2.compute import generate_tregenza_patches

PDAL_BIN = "/usr/bin/pdal"
SITE_KEY = "vidigal"
DEFAULT_DATA_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

#: Vertical band around the probe's measured TLS envelope (108.66-185.89 m,
#: runs/wp03_tls_probe_20260908/tls_georeference_probe.json) -- wide enough to
#: keep every real return, tight enough to drop stray outliers.
E57_Z_RANGE = (50.0, 250.0)

CELL_M_SENSITIVITY = 0.5  # spec 1c: TLS DSM sensitivity row
GROUND_CLEARANCE_M = 2.0  # spec 1b: registration cells must sit this far from any footprint
ALLEY_CLASSES = ("<1.5m", "1.5-3m", ">3m")  # spec 1c, narrowest first
G2_TOLERANCE = 0.10  # spec 1c: the gate's median |delta| <= 0.10 criterion
G2_MIN_POINTS_PER_CLASS = 300  # spec 1c
FACADE_MAX_POINTS = 2000  # spec 1d
#: Storey bins reused verbatim from the WP-04F facade convention
#: (runs/wp04f_facade_20260915T060126Z/comparison.md), not invented here.
STOREY_BIN_EDGES = (0.0, 3.0, 6.0, 9.0)
STOREY_BIN_LABELS = ("0-3m", "3-6m", "6-9m", ">9m")

MAX_PDAL_WALL_SECONDS = 40 * 60  # spec: fall back to 1 m DSM only past ~40 minutes total

#: WP-03B deliverable 4 (docs/wp03b_tls_diagnostic_spec.md): "filters.smrf if it
#: runs in < 10 min on the 1 m grid".
GROUND_DEF_MAX_WALL_SECONDS = 10 * 60
#: WP-03B deliverable 2c coverage-share observer filter.
COVERAGE_MIN_SHARE = 0.8


# ---------------------------------------------------------------------------
# PDAL: TLS DSM / ground raster from the three e57 scans
# ---------------------------------------------------------------------------

def e57_scan_paths(data_root: Path) -> list[Path]:
    d = data_root / "data" / "vidigal_tls" / "raw" / "pointclouds" / "Nuvens Separadas"
    return sorted(d.glob("*-registered.e57"))


def build_scan_pipeline(e57_path: Path, out_tif: Path, cell_m: float, srs: str = "EPSG:31983") -> list[dict]:
    """readers.e57 -> filters.range(Z) -> filters.stats(Z) -> one writers.gdal stage.

    `output_type="min,max,count"` puts all three statistics in one raster
    (bands 1/2/3) from a SINGLE point-cloud read -- PDAL's `pdal pipeline`
    executor does not fan a filter out to more than one writers.gdal stage
    (confirmed empirically 2026-09-15: a second writer sharing the same
    `inputs` tag silently never ran), so one resolution == one read.
    `override_srs` is needed because readers.e57 does not itself carry a CRS
    (probe: coordinates are EPSG:31983 project coordinates, "registered" in
    the filename means georeferenced, not that the driver tags the SRS).
    """
    return [
        {"type": "readers.e57", "filename": str(e57_path), "override_srs": srs},
        {"type": "filters.range", "limits": f"Z[{E57_Z_RANGE[0]}:{E57_Z_RANGE[1]}]"},
        {"type": "filters.stats", "dimensions": "Z"},
        {
            "type": "writers.gdal",
            "filename": str(out_tif),
            "resolution": cell_m,
            "output_type": "min,max,count",
            "gdaldriver": "GTiff",
            "nodata": -9999,
        },
    ]


def run_pdal_pipeline(pipeline: list[dict], pipeline_json_path: Path, metadata_path: Path) -> dict:
    pipeline_json_path.write_text(json.dumps(pipeline, indent=1))
    t0 = _time.monotonic()
    proc = subprocess.run(
        [PDAL_BIN, "pipeline", str(pipeline_json_path), "--metadata", str(metadata_path)],
        capture_output=True, text=True,
    )
    elapsed_s = _time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"pdal pipeline failed ({pipeline_json_path.name}): {proc.stderr}")
    meta = json.loads(metadata_path.read_text())
    stats = meta.get("stages", {}).get("filters.stats", {}).get("statistic", [])
    point_count = int(stats[0]["count"]) if stats else None
    return {"elapsed_s": elapsed_s, "point_count": point_count}


def _pdal_version() -> str:
    proc = subprocess.run([PDAL_BIN, "--version"], capture_output=True, text=True)
    for line in (proc.stdout or proc.stderr).splitlines():
        line = line.strip()
        if line.startswith("pdal"):
            return line
    return "unknown"


def build_tls_rasters(data_root: Path, tmp_dir: Path, max_wall_seconds: float = MAX_PDAL_WALL_SECONDS) -> dict:
    """Run PDAL over the 3 Vidigal e57 scans, merge per-scan rasters, return paths + stats.

    1 m (DSM + ground) always runs. 0.5 m DSM (spec's sensitivity row) only
    runs if the 1 m pass leaves enough of `max_wall_seconds`; if not, this
    returns with `dsm_0p5m=None` and `fallback_reason` set (spec: "fall back
    to a 1 m DSM only and say so").
    """
    scans = e57_scan_paths(data_root)
    if len(scans) == 0:
        raise FileNotFoundError(f"no *-registered.e57 under {data_root}")

    pipelines_used = {}
    point_counts = {}
    per_scan_elapsed = {}
    t_start = _time.monotonic()

    def _run_resolution(cell_m: float, tag: str) -> list[Path]:
        out_tifs = []
        for scan in scans:
            out_tif = tmp_dir / f"{scan.stem}_{tag}.tif"
            pipeline = build_scan_pipeline(scan, out_tif, cell_m)
            pjson = tmp_dir / f"{scan.stem}_{tag}_pipeline.json"
            meta_json = tmp_dir / f"{scan.stem}_{tag}_meta.json"
            result = run_pdal_pipeline(pipeline, pjson, meta_json)
            pipelines_used[f"{scan.stem}_{tag}"] = pipeline
            point_counts[f"{scan.stem}_{tag}"] = result["point_count"]
            per_scan_elapsed[f"{scan.stem}_{tag}"] = result["elapsed_s"]
            out_tifs.append(out_tif)
        return out_tifs

    tifs_1m = _run_resolution(CELL_M, "1m")
    elapsed_after_1m = _time.monotonic() - t_start

    fallback_reason = None
    tifs_0p5m: list[Path] = []
    remaining = max_wall_seconds - elapsed_after_1m
    if remaining > elapsed_after_1m * 0.5:
        tifs_0p5m = _run_resolution(CELL_M_SENSITIVITY, "0p5m")
    else:
        fallback_reason = (
            f"1 m pass used {elapsed_after_1m:.0f}s of the {max_wall_seconds:.0f}s budget; "
            "skipped the 0.5 m sensitivity pass rather than risk the budget."
        )

    total_elapsed = _time.monotonic() - t_start

    dsm_1m, ground_1m, count_1m, meta_1m = _merge_bands(tifs_1m)
    dsm_0p5m = None
    grid_0p5m = None
    if tifs_0p5m:
        dsm_0p5m, _ground_0p5m, _count_0p5m, grid_0p5m = _merge_bands(tifs_0p5m)

    return {
        "dsm_1m": dsm_1m, "ground_1m": ground_1m, "count_1m": count_1m, "grid_1m": meta_1m,
        "dsm_0p5m": dsm_0p5m, "grid_0p5m": grid_0p5m,
        "scans": [str(s) for s in scans],
        "point_counts": point_counts,
        "per_scan_elapsed_s": per_scan_elapsed,
        "elapsed_1m_s": elapsed_after_1m,
        "total_elapsed_s": total_elapsed,
        "fallback_reason": fallback_reason,
        "pipelines": pipelines_used,
    }


def _merge_bands(per_scan_tifs: list[Path]):
    """Merge per-scan (min, max, count) rasters into one grid: max-of-maxes,
    min-of-mins, sum-of-counts. Per-scan writers.gdal auto-sizes each raster
    to its own point extent, so this is not a same-grid stack -- rasterio's
    merge() builds the output grid from the union of bounds at the shared
    resolution and reads each source through its own transform."""
    srcs = [rasterio.open(t) for t in per_scan_tifs]
    try:
        ground_arr, out_transform = rio_merge(srcs, method="min", indexes=[1], nodata=-9999.0)
        dsm_arr, _t = rio_merge(srcs, method="max", indexes=[2], nodata=-9999.0)
        count_arr, _t = rio_merge(srcs, method="sum", indexes=[3], nodata=-9999.0)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    dsm_arr = dsm_arr[0].astype("float32")
    ground_arr = ground_arr[0].astype("float32")
    count_arr = count_arr[0].astype("float32")
    meta = {"transform": out_transform, "crs": crs, "shape": dsm_arr.shape}
    return dsm_arr, ground_arr, count_arr, meta


def _write_tif(path: Path, arr: np.ndarray, transform, crs, nodata: float = -9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(np.nan_to_num(arr, nan=nodata).astype("float32"), 1)


# ---------------------------------------------------------------------------
# Registration residual (replaces the missing .rcp floor)
# ---------------------------------------------------------------------------

def _resample_to_grid(src_path: Path, dst_transform, dst_shape, dst_crs) -> np.ndarray:
    out = np.full(dst_shape, np.nan, dtype="float64")
    with rasterio.open(src_path) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return out


def _resample_array_to_grid(
    src_arr: np.ndarray, src_transform, src_crs,
    dst_transform, dst_shape, dst_crs,
    resampling=rasterio.warp.Resampling.bilinear,
) -> np.ndarray:
    """Same job as `_resample_to_grid` but from an in-memory array (the ALS
    2.5D surface, already built by wp04_sites.build_site_surface) instead of
    a file on disk."""
    out = np.full(dst_shape, np.nan, dtype="float64")
    rasterio.warp.reproject(
        source=np.asarray(src_arr, dtype="float64"), destination=out,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=resampling, dst_nodata=np.nan,
    )
    return out


def xy_to_rowcol(transform, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    cols = np.floor(inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c).astype(int)
    rows = np.floor(inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f).astype(int)
    return rows, cols


def _shift_transform(transform, dx_cells: float, dy_cells: float):
    """Origin-shifted transform: the returned transform's pixel (0,0) sits at
    `transform`'s (pixel) location (dx_cells, dy_cells) -- deliverable 3's
    "shift the TLS rasters by (dx, dy)" and "±0.5-cell origin correction",
    both expressed as the same operation at different magnitudes."""
    return transform * Affine.translation(dx_cells, dy_cells)


def _clearance_mask(footprints: gpd.GeoDataFrame, transform, shape, clearance_m: float) -> np.ndarray:
    building_mask = rasterize(
        [(geom, 1) for geom in footprints.geometry if geom is not None and not geom.is_empty],
        out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    cell_m = abs(transform.a)
    dist = ndimage.distance_transform_edt(~building_mask) * cell_m
    return dist >= clearance_m


def _shifted_overlap(a: np.ndarray, b: np.ndarray, dy: int, dx: int):
    h, w = a.shape
    ay0, ay1 = max(0, dy), h + min(0, dy)
    ax0, ax1 = max(0, dx), w + min(0, dx)
    by0, by1 = max(0, -dy), h + min(0, -dy)
    bx0, bx1 = max(0, -dx), w + min(0, -dx)
    return a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1]


def planar_shift(a: np.ndarray, b: np.ndarray, search_cells: int = 5, min_overlap: int = 100) -> dict:
    """Integer-pixel (dy, dx) grid-search cross-correlation: the shift of `b`
    relative to `a` that maximises the Pearson r over their valid overlap.
    Grid search (not FFT phase correlation) because the expected shift is
    "<= 1 cell" (spec) -- a +/-`search_cells` window is already generous."""
    best_dy, best_dx, best_r = 0, 0, -np.inf
    for dy in range(-search_cells, search_cells + 1):
        for dx in range(-search_cells, search_cells + 1):
            av, bv = _shifted_overlap(a, b, dy, dx)
            if av.size == 0:
                continue
            m = np.isfinite(av) & np.isfinite(bv)
            if m.sum() < min_overlap:
                continue
            av_v, bv_v = av[m], bv[m]
            if av_v.std() == 0 or bv_v.std() == 0:
                continue
            r = float(np.corrcoef(av_v, bv_v)[0, 1])
            if r > best_r:
                best_dy, best_dx, best_r = dy, dx, r
    return {"dy_cells": best_dy, "dx_cells": best_dx, "r": None if best_r == -np.inf else best_r}


def registration_residual(
    tls_ground: np.ndarray, tls_transform, tls_crs,
    dtm_path: Path, footprints: gpd.GeoDataFrame,
    clearance_m: float = GROUND_CLEARANCE_M,
) -> dict:
    shape = tls_ground.shape
    dtm_on_tls = _resample_to_grid(dtm_path, tls_transform, shape, tls_crs)
    clearance = _clearance_mask(footprints, tls_transform, shape, clearance_m)

    tls_valid = tls_ground != -9999.0
    valid = tls_valid & clearance & np.isfinite(dtm_on_tls)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "median_delta_m": None, "p95_abs_delta_m": None, "shift": None, "clearance_m": clearance_m}

    delta = tls_ground[valid].astype("float64") - dtm_on_tls[valid]
    median_delta = float(np.median(delta))
    p95_abs = float(np.percentile(np.abs(delta), 95))

    a = np.where(tls_valid, tls_ground.astype("float64"), np.nan)
    b = np.where(np.isfinite(dtm_on_tls), dtm_on_tls, np.nan)
    shift = planar_shift(a, b)

    result = {
        "n": n, "clearance_m": clearance_m,
        "median_delta_m": median_delta, "p95_abs_delta_m": p95_abs,
        "shift": shift,
        "corrected": False,
    }
    if shift["dy_cells"] != 0 or shift["dx_cells"] != 0:
        cell_m = abs(tls_transform.a)
        bv_shift, av_shift = _shifted_overlap(b, a, -shift["dy_cells"], -shift["dx_cells"])
        m = np.isfinite(av_shift) & np.isfinite(bv_shift)
        delta_corrected = av_shift[m] - bv_shift[m]
        result["shift_applied"] = False  # spec: report; don't correct unless |shift| >= 1 cell, and keep both if so
        result["corrected_variant"] = {
            "n": int(m.sum()),
            "median_delta_m": float(np.median(delta_corrected)) if m.sum() else None,
            "p95_abs_delta_m": float(np.percentile(np.abs(delta_corrected), 95)) if m.sum() else None,
            "shift_m": (shift["dy_cells"] * cell_m, shift["dx_cells"] * cell_m),
        }
    return result


def registration_residual_at_shift(
    tls_ground: np.ndarray, tls_transform, tls_crs, dtm_path: Path,
    footprints: gpd.GeoDataFrame, dx_cells: float, dy_cells: float,
    clearance_m: float = GROUND_CLEARANCE_M,
) -> dict:
    """WP-03B deliverable 3: the registration_residual computation, replayed
    with the TLS grid's origin shifted by (dx_cells, dy_cells) pixels before
    resampling the ALS DTM onto it -- tests one specific (dx, dy) hypothesis
    (the detected integer planar shift, or a ±0.5-cell origin convention
    mismatch between rasterio.merge's output grid and the ALS raster) in
    isolation, without touching the unshifted rasters `registration_residual`
    itself reports."""
    shifted = _shift_transform(tls_transform, dx_cells, dy_cells)
    shape = tls_ground.shape
    dtm_on_tls = _resample_to_grid(dtm_path, shifted, shape, tls_crs)
    clearance = _clearance_mask(footprints, shifted, shape, clearance_m)
    tls_valid = tls_ground != -9999.0
    valid = tls_valid & clearance & np.isfinite(dtm_on_tls)
    n = int(valid.sum())
    if n == 0:
        return {"dx_cells": dx_cells, "dy_cells": dy_cells, "n": 0, "median_delta_m": None, "p95_abs_delta_m": None}
    delta = tls_ground[valid].astype("float64") - dtm_on_tls[valid]
    return {
        "dx_cells": dx_cells, "dy_cells": dy_cells, "n": n,
        "median_delta_m": float(np.median(delta)),
        "p95_abs_delta_m": float(np.percentile(np.abs(delta), 95)),
    }


def shift_check(tls: dict, footprints: gpd.GeoDataFrame, data_root: Path, registration: dict) -> dict:
    """WP-03B deliverable 3(i): baseline vs detected-shift vs ±0.5-cell origin
    correction, side by side. `registration` is the same-run's
    `registration_residual(...)` output -- its `shift` field supplies the
    detected (dx, dy); the half-cell probe is scoped to x only because
    `planar_shift`'s grid search (docstring: wp03_tls.py) found dy_cells=0 --
    a y-axis half-cell test would be probing a dimension with no detected
    signal."""
    tls_ground = tls["ground_1m"]
    transform = tls["grid_1m"]["transform"]
    crs = tls["grid_1m"]["crs"]
    dtm_path = data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif"

    detected_dx = float(registration["shift"]["dx_cells"])
    detected_dy = float(registration["shift"]["dy_cells"])

    baseline = registration_residual_at_shift(tls_ground, transform, crs, dtm_path, footprints, 0.0, 0.0)
    detected = registration_residual_at_shift(tls_ground, transform, crs, dtm_path, footprints, detected_dx, detected_dy)
    origin_plus = registration_residual_at_shift(tls_ground, transform, crs, dtm_path, footprints, 0.5, 0.0)
    origin_minus = registration_residual_at_shift(tls_ground, transform, crs, dtm_path, footprints, -0.5, 0.0)

    medians = {
        "baseline_unshifted": baseline["median_delta_m"],
        "detected_shift": detected["median_delta_m"],
        "origin_plus_0p5_cell_x": origin_plus["median_delta_m"],
        "origin_minus_0p5_cell_x": origin_minus["median_delta_m"],
    }
    finite = {k: v for k, v in medians.items() if v is not None}
    closest_to_zero = min(finite, key=lambda k: abs(finite[k])) if finite else None

    return {
        "detected": {"dx_cells": detected_dx, "dy_cells": detected_dy},
        "baseline": baseline,
        "detected_shift": detected,
        "origin_plus_0p5_cell_x": origin_plus,
        "origin_minus_0p5_cell_x": origin_minus,
        "medians_side_by_side": medians,
        "closest_to_zero": closest_to_zero,
        "note": "half-cell test scoped to the x axis only -- the detected planar shift has dy_cells=0.",
    }


# ---------------------------------------------------------------------------
# G2: alley-width validity floor
# ---------------------------------------------------------------------------

def alley_width_class(xy: np.ndarray, footprints: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) to the nearest footprint edge, and its alley-width class.

    Spec 1c: width = 2 x distance to the nearest footprint edge (the point
    sits on the alley centreline; twice the gap to the flanking wall
    estimates the open-space width between the two buildings).
    """
    geoms = footprints.geometry.to_numpy()
    tree = STRtree(geoms)
    pts = np.array([Point(x, y) for x, y in xy], dtype=object)
    n = len(pts)
    if n == 0:
        return np.zeros(0), np.array([], dtype=object)
    nearest_idx = np.atleast_1d(tree.nearest(pts))
    dist = np.array([pts[i].distance(geoms[nearest_idx[i]]) for i in range(n)])
    width = 2.0 * dist
    labels = np.select(
        [width < 1.5, width < 3.0],
        [ALLEY_CLASSES[0], ALLEY_CLASSES[1]],
        default=ALLEY_CLASSES[2],
    )
    return dist, labels


def g2_floor(class_table: list[dict]) -> str | None:
    """Narrowest class whose median |delta| <= G2_TOLERANCE, else None.

    `class_table` entries need "class" and "median_abs_delta" keys.
    Ordered by ALLEY_CLASSES (narrowest first) regardless of input order.
    """
    by_class = {row["class"]: row for row in class_table}
    for cls in ALLEY_CLASSES:
        row = by_class.get(cls)
        if row is not None and row["median_abs_delta"] is not None and row["median_abs_delta"] <= G2_TOLERANCE:
            return cls
    return None


def compute_svf_pair(
    surface_a: np.ndarray, transform_a, cell_a: float,
    surface_b: np.ndarray, transform_b, cell_b: float,
    obs_xy: np.ndarray, directions: np.ndarray, weights: np.ndarray,
    obs_height_m: float, max_dist_m: float, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """SVF (solid-angle-weighted, geometric -- no cumulative-sky/kWh weighting
    so this does not depend on the EPW-derived weather sky) at the SAME
    observer points on two independently-marched surfaces."""
    vis_a, _ob_a = patch_visibility(
        surface_a, transform_a, obs_xy, directions=directions,
        obs_height_m=obs_height_m, max_dist_m=max_dist_m, step_m=cell_a, device=device,
    )
    vis_b, _ob_b = patch_visibility(
        surface_b, transform_b, obs_xy, directions=directions,
        obs_height_m=obs_height_m, max_dist_m=max_dist_m, step_m=cell_b, device=device,
    )
    svf_a = svf_solid_angle(vis_a, weights)
    svf_b = svf_solid_angle(vis_b, weights)
    return svf_a, svf_b


def g2_candidate_points(als_surface, als_transform, als_is_building, tls_dsm, tls_transform, tls_crs) -> np.ndarray:
    """Every ALS-grid ground cell whose TLS DSM cell (nearest, <= 1 m) is populated."""
    h, w = als_surface.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs, ys = rasterio.transform.xy(als_transform, rows.ravel(), cols.ravel(), offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    ground_mask = ~als_is_building.ravel() if als_is_building is not None else np.ones(xs.shape, dtype=bool)

    inv = ~tls_transform
    cols_tls = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
    rows_tls = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
    th, tw = tls_dsm.shape
    in_bounds = (rows_tls >= 0) & (rows_tls < th) & (cols_tls >= 0) & (cols_tls < tw)
    covered = np.zeros(xs.shape, dtype=bool)
    covered[in_bounds] = tls_dsm[rows_tls[in_bounds], cols_tls[in_bounds]] != -9999.0

    keep = ground_mask & covered
    return np.column_stack([xs[keep], ys[keep]])


# ---------------------------------------------------------------------------
# WP-03B deliverable 1: coverage diagnostic
# ---------------------------------------------------------------------------

def coverage_diagnostic(
    tls_dsm: np.ndarray, tls_transform, tls_crs, footprints: gpd.GeoDataFrame,
) -> tuple[dict, np.ndarray]:
    """Share of TLS-uncovered cells that lie inside a 2019 footprint vs on
    open ground -- confirms or kills the wp03b_tls_diagnostic_spec hypothesis
    that wp03_tls.py:415's DTM fill deletes buildings from the TLS surface
    (a ground-level scanner in alleys sees facades and little of the roofs,
    so its uncovered cells should be overwhelmingly BUILDING cells, not
    ground, if that hypothesis is right)."""
    covered = tls_dsm != -9999.0
    building_mask = rasterize(
        [(geom, 1) for geom in footprints.geometry if geom is not None and not geom.is_empty],
        out_shape=tls_dsm.shape, transform=tls_transform, fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)

    uncovered = ~covered
    n_total = int(uncovered.size)
    n_uncovered = int(uncovered.sum())
    n_unc_fp = int((uncovered & building_mask).sum())
    n_unc_ground = int((uncovered & ~building_mask).sum())

    stats = {
        "n_cells_total": n_total,
        "n_covered": int(covered.sum()),
        "n_uncovered": n_uncovered,
        "share_uncovered_of_total": (n_uncovered / n_total) if n_total else None,
        "n_uncovered_inside_footprint": n_unc_fp,
        "n_uncovered_on_ground": n_unc_ground,
        "share_uncovered_inside_footprint": (n_unc_fp / n_uncovered) if n_uncovered else None,
        "share_uncovered_on_ground": (n_unc_ground / n_uncovered) if n_uncovered else None,
    }
    return stats, building_mask


def write_coverage_map_png(path: Path, tls_dsm: np.ndarray, building_mask: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    covered = tls_dsm != -9999.0
    rgb = np.empty(covered.shape + (3,), dtype="float32")
    rgb[covered & ~building_mask] = (0.20, 0.60, 0.20)   # covered, ground
    rgb[covered & building_mask] = (0.20, 0.40, 0.80)    # covered, building (rooftop returns)
    rgb[~covered & building_mask] = (0.80, 0.20, 0.20)   # UNCOVERED inside footprint (the suspect class)
    rgb[~covered & ~building_mask] = (0.90, 0.90, 0.70)  # uncovered, on open ground

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin="upper")
    ax.set_title(
        "TLS coverage vs 2019 footprints (Vidigal)\n"
        "green=covered ground  blue=covered building  red=UNCOVERED-inside-footprint  tan=uncovered-ground"
    )
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# WP-03B deliverable 2: ALS-fill surface + covered-only observer filter
# ---------------------------------------------------------------------------

def als_fill_surface(tls_dsm: np.ndarray, tls_transform, tls_crs, als_surface: np.ndarray, als_transform, als_crs) -> np.ndarray:
    """Variant (b): uncovered TLS cells take the ALS 2.5D surface value
    (buildings included) instead of the ALS DTM -- only TLS-COVERED cells can
    now differ between the two surfaces the G2 comparison marches, so a gap
    can no longer be explained by the fill deleting buildings. Nearest-
    neighbour, not bilinear: the ALS surface has sharp building-edge
    discontinuities that bilinear resampling would blur into fictitious
    partial-height steps right at the footprint boundary -- the opposite of
    what this variant is trying to test."""
    als_on_tls = _resample_array_to_grid(
        als_surface, als_transform, als_crs, tls_transform, tls_dsm.shape, tls_crs,
        resampling=rasterio.warp.Resampling.nearest,
    )
    return np.where(tls_dsm != -9999.0, tls_dsm, als_on_tls).astype("float32")


def coverage_share_disc(covered: np.ndarray, cell_m: float, max_dist_m: float) -> np.ndarray:
    """Per-cell share of TLS-covered cells within a `max_dist_m` disc (the
    same radius patch_visibility marches to) via FFT disc convolution. The
    denominator is bounded by the TLS raster's own array extent -- cells
    outside it are outside the known domain, not simply "uncovered"."""
    radius_cells = max_dist_m / cell_m
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = ((yy ** 2 + xx ** 2) <= radius_cells ** 2).astype("float32")

    covered_f = covered.astype("float32")
    domain_f = np.ones_like(covered_f)
    num = fftconvolve(covered_f, disc, mode="same")
    den = fftconvolve(domain_f, disc, mode="same")
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)


def coverage_share_mask(obs_xy: np.ndarray, tls_transform, coverage_share: np.ndarray, min_share: float) -> np.ndarray:
    """Variant (c): keep exactly the observers whose march disc meets
    `min_share` TLS coverage, per `coverage_share_disc`."""
    rows, cols = xy_to_rowcol(tls_transform, obs_xy)
    h, w = coverage_share.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    keep = np.zeros(len(obs_xy), dtype=bool)
    keep[in_bounds] = coverage_share[rows[in_bounds], cols[in_bounds]] >= min_share
    return keep


def compute_g2_variant(
    footprints: gpd.GeoDataFrame, als_bundle,
    tls_dsm: np.ndarray, tls_transform, tls_crs,
    tls_filled: np.ndarray, tls_filled_transform,
    device: str, variant: str,
    coverage_share: np.ndarray | None = None, coverage_min_share: float = COVERAGE_MIN_SHARE,
) -> dict:
    """One G2 class table for one surface variant. `tls_dsm`/`tls_transform`
    define TLS coverage (and so the candidate-observer set, via
    `g2_candidate_points` -- unchanged from phase 1); `tls_filled`/
    `tls_filled_transform` is the surface actually marched for SVF, which is
    what differs per variant."""
    directions, weights = generate_tregenza_patches()
    als_surface, als_transform, als_crs, als_is_building = als_bundle[:4]

    obs_xy = g2_candidate_points(als_surface, als_transform, als_is_building, tls_dsm, tls_transform, tls_crs)
    dist, labels = alley_width_class(obs_xy, footprints)
    n_candidates = len(obs_xy)

    if coverage_share is not None:
        keep = coverage_share_mask(obs_xy, tls_transform, coverage_share, coverage_min_share)
        obs_xy, labels = obs_xy[keep], labels[keep]

    rows = []
    for cls in ALLEY_CLASSES:
        idx = np.where(labels == cls)[0]
        n_cls = len(idx)
        row = {"class": cls, "n": n_cls}
        if n_cls == 0:
            row.update(r=None, median_delta=None, median_abs_delta=None, p95_abs_delta=None, share_within_tol=None)
            rows.append(row)
            continue
        xy_cls = obs_xy[idx]
        svf_als, svf_tls = compute_svf_pair(
            als_surface, als_transform, CELL_M,
            tls_filled, tls_filled_transform, CELL_M,
            xy_cls, directions, weights, OBS_HEIGHT_M, MAX_DIST_M, device,
        )
        delta = svf_als - svf_tls
        r = float(np.corrcoef(svf_als, svf_tls)[0, 1]) if n_cls > 1 and svf_als.std() > 0 and svf_tls.std() > 0 else None
        row.update(
            r=r,
            median_delta=float(np.median(delta)),
            median_abs_delta=float(np.median(np.abs(delta))),
            p95_abs_delta=float(np.percentile(np.abs(delta), 95)),
            share_within_tol=float(np.mean(np.abs(delta) <= G2_TOLERANCE)),
            below_min_points=n_cls < G2_MIN_POINTS_PER_CLASS,
        )
        rows.append(row)

    return {
        "variant": variant, "cell_m": CELL_M,
        "n_candidates": n_candidates, "n_used": len(obs_xy),
        "classes": rows, "floor": g2_floor(rows),
    }


def compute_g2_variants(
    data_root: Path, tls: dict, footprints: gpd.GeoDataFrame, als_bundle, device: str,
    coverage_min_share: float = COVERAGE_MIN_SHARE,
) -> dict:
    """Deliverable 2, variants (a)-(c): same engine, same observer-selection
    rule as phase 1, reused via `compute_g2_variant`."""
    als_surface, als_transform, als_crs, _als_is_building = als_bundle[:4]
    tls_dsm = tls["dsm_1m"]
    tls_transform = tls["grid_1m"]["transform"]
    tls_crs = tls["grid_1m"]["crs"]

    dtm_on_tls = _resample_to_grid(
        data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", tls_transform, tls_dsm.shape, tls_crs
    )
    filled_a = np.where(tls_dsm != -9999.0, tls_dsm, dtm_on_tls).astype("float32")
    variant_a = compute_g2_variant(
        footprints, als_bundle, tls_dsm, tls_transform, tls_crs, filled_a, tls_transform, device, "a_dtm_fill_merged"
    )

    filled_b = als_fill_surface(tls_dsm, tls_transform, tls_crs, als_surface, als_transform, als_crs)
    variant_b = compute_g2_variant(
        footprints, als_bundle, tls_dsm, tls_transform, tls_crs, filled_b, tls_transform, device, "b_als_fill"
    )

    covered = tls_dsm != -9999.0
    coverage_share = coverage_share_disc(covered, CELL_M, MAX_DIST_M)
    variant_c = compute_g2_variant(
        footprints, als_bundle, tls_dsm, tls_transform, tls_crs, filled_b, tls_transform, device,
        "c_covered_only_observers", coverage_share=coverage_share, coverage_min_share=coverage_min_share,
    )

    return {
        "a_dtm_fill_merged": variant_a, "b_als_fill": variant_b, "c_covered_only_observers": variant_c,
        "coverage_min_share": coverage_min_share,
        "coverage_share_note": (
            f"coverage_share_disc uses max_dist_m={MAX_DIST_M:g} m (the same radius patch_visibility "
            f"marches to), which exceeds the TLS raster's own extent ({tls_dsm.shape[0]}x{tls_dsm.shape[1]} "
            f"cells at {CELL_M:g} m) -- the disc is effectively the whole raster for nearly every "
            "observer, so the per-observer share is close to the raster's global coverage fraction "
            "everywhere, and the >=80% filter can retain very few or zero observers."
        ),
    }


def compute_g2(
    data_root: Path, tls: dict, footprints: gpd.GeoDataFrame, als_bundle,
    device: str, rng: np.random.Generator,
) -> dict:
    directions, weights = generate_tregenza_patches()
    als_surface, als_transform, als_crs, als_is_building = als_bundle[:4]
    dtm_on_tls = _resample_to_grid(data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", tls["grid_1m"]["transform"], tls["dsm_1m"].shape, tls["grid_1m"]["crs"])
    tls_dsm_filled = np.where(tls["dsm_1m"] != -9999.0, tls["dsm_1m"], dtm_on_tls).astype("float32")

    obs_xy = g2_candidate_points(
        als_surface, als_transform, als_is_building, tls["dsm_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"]
    )
    dist, labels = alley_width_class(obs_xy, footprints)

    rows = []
    for cls in ALLEY_CLASSES:
        cls_mask = labels == cls
        idx = np.where(cls_mask)[0]
        n_cls = len(idx)
        row = {"class": cls, "n": n_cls}
        if n_cls == 0:
            row.update(r=None, median_delta=None, median_abs_delta=None, p95_abs_delta=None, share_within_tol=None)
            rows.append(row)
            continue
        xy_cls = obs_xy[idx]
        svf_als, svf_tls = compute_svf_pair(
            als_surface, als_transform, CELL_M,
            tls_dsm_filled, tls["grid_1m"]["transform"], CELL_M,
            xy_cls, directions, weights, OBS_HEIGHT_M, MAX_DIST_M, device,
        )
        delta = svf_als - svf_tls
        r = float(np.corrcoef(svf_als, svf_tls)[0, 1]) if n_cls > 1 and svf_als.std() > 0 and svf_tls.std() > 0 else None
        row.update(
            r=r,
            median_delta=float(np.median(delta)),
            median_abs_delta=float(np.median(np.abs(delta))),
            p95_abs_delta=float(np.percentile(np.abs(delta), 95)),
            share_within_tol=float(np.mean(np.abs(delta) <= G2_TOLERANCE)),
            below_min_points=n_cls < G2_MIN_POINTS_PER_CLASS,
        )
        rows.append(row)

    result = {"cell_m": CELL_M, "classes": rows, "floor": g2_floor(rows)}

    if tls["dsm_0p5m"] is not None:
        grid_05 = tls["grid_0p5m"]
        dtm_on_tls_05 = _resample_to_grid(
            data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", grid_05["transform"], tls["dsm_0p5m"].shape, grid_05["crs"]
        )
        tls_05_filled = np.where(tls["dsm_0p5m"] != -9999.0, tls["dsm_0p5m"], dtm_on_tls_05).astype("float32")
        rows_05 = []
        for cls in ALLEY_CLASSES:
            idx = np.where(labels == cls)[0]
            n_cls = len(idx)
            row = {"class": cls, "n": n_cls}
            if n_cls == 0:
                row.update(median_abs_delta=None, p95_abs_delta=None)
                rows_05.append(row)
                continue
            xy_cls = obs_xy[idx]
            svf_als, svf_tls05 = compute_svf_pair(
                als_surface, als_transform, CELL_M,
                tls_05_filled, grid_05["transform"], CELL_M_SENSITIVITY,
                xy_cls, directions, weights, OBS_HEIGHT_M, MAX_DIST_M, device,
            )
            delta = svf_als - svf_tls05
            row.update(median_abs_delta=float(np.median(np.abs(delta))), p95_abs_delta=float(np.percentile(np.abs(delta), 95)))
            rows_05.append(row)
        result["sensitivity_0p5m"] = {"cell_m": CELL_M_SENSITIVITY, "classes": rows_05}

    return result


# ---------------------------------------------------------------------------
# WP-03B deliverable 4: ground definition check (min vs a robust per-cell ground)
# ---------------------------------------------------------------------------

def _run_pdal_pipeline_timeout(pipeline: list[dict], pipeline_json_path: Path, timeout_s: float) -> bool:
    """Same job as `run_pdal_pipeline` but bounded by a wall-clock timeout
    instead of running to completion -- SMRF over a 100M+ point scan can run
    long, and this check has its own (smaller) spec budget than the main PDAL
    extraction."""
    pipeline_json_path.write_text(json.dumps(pipeline, indent=1))
    try:
        proc = subprocess.run(
            [PDAL_BIN, "pipeline", str(pipeline_json_path)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def build_smrf_ground_pipeline(e57_path: Path, out_tif: Path, cell_m: float, srs: str = "EPSG:31983") -> list[dict]:
    """Deliverable 4's offered fallback ('filters.smrf if it runs in < 10 min
    on the 1 m grid'): PDAL's writers.gdal has no per-cell-percentile
    output_type, and extracting the ~262M raw points into Python to compute a
    literal numpy 5th-percentile-per-cell is not tractable in this check's
    time budget -- SMRF's morphological ground classification is the spec's
    own substitute, reported as SMRF, not relabelled as a percentile."""
    return [
        {"type": "readers.e57", "filename": str(e57_path), "override_srs": srs},
        {"type": "filters.range", "limits": f"Z[{E57_Z_RANGE[0]}:{E57_Z_RANGE[1]}]"},
        {"type": "filters.smrf"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {
            "type": "writers.gdal",
            "filename": str(out_tif),
            "resolution": cell_m,
            "output_type": "min",
            "gdaldriver": "GTiff",
            "nodata": -9999,
        },
    ]


def build_smrf_ground_raster(data_root: Path, tmp_dir: Path, max_wall_seconds: float = GROUND_DEF_MAX_WALL_SECONDS) -> dict:
    scans = e57_scan_paths(data_root)
    t0 = _time.monotonic()
    out_tifs: list[Path] = []
    fallback_reason = None

    for scan in scans:
        remaining = max_wall_seconds - (_time.monotonic() - t0)
        if remaining <= 0:
            fallback_reason = (
                f"SMRF ground pass used the full {max_wall_seconds:.0f}s budget after "
                f"{len(out_tifs)}/{len(scans)} scans; stopped rather than overrun."
            )
            break
        out_tif = tmp_dir / f"{scan.stem}_smrf_ground.tif"
        pjson = tmp_dir / f"{scan.stem}_smrf_pipeline.json"
        pipeline = build_smrf_ground_pipeline(scan, out_tif, CELL_M)
        ok = _run_pdal_pipeline_timeout(pipeline, pjson, remaining)
        if not ok:
            fallback_reason = (
                f"SMRF pipeline on {scan.name} did not finish within the remaining "
                f"{remaining:.0f}s of the {max_wall_seconds:.0f}s budget."
            )
            break
        out_tifs.append(out_tif)

    total_elapsed = _time.monotonic() - t0
    if len(out_tifs) < len(scans):
        return {"ground_smrf": None, "grid": None, "elapsed_s": total_elapsed, "fallback_reason": fallback_reason or "incomplete"}

    srcs = [rasterio.open(t) for t in out_tifs]
    try:
        arr, out_transform = rio_merge(srcs, method="min", indexes=[1], nodata=-9999.0)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    arr = arr[0].astype("float32")
    return {
        "ground_smrf": arr, "grid": {"transform": out_transform, "crs": crs, "shape": arr.shape},
        "elapsed_s": total_elapsed, "fallback_reason": None,
    }


def ground_definition_check(
    data_root: Path, tmp_dir: Path, tls: dict, footprints: gpd.GeoDataFrame, run_dir: Path,
    max_wall_seconds: float = GROUND_DEF_MAX_WALL_SECONDS,
) -> dict:
    """Deliverable 4: replace `min` ground with a robust per-cell ground
    estimate (SMRF classification, see `build_smrf_ground_pipeline`) and
    report the registration residual under both."""
    dtm_path = data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif"
    min_residual = registration_residual(
        tls["ground_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"], dtm_path, footprints,
    )
    min_ground = {
        "n": min_residual["n"], "median_delta_m": min_residual["median_delta_m"],
        "p95_abs_delta_m": min_residual["p95_abs_delta_m"],
    }

    smrf = build_smrf_ground_raster(data_root, tmp_dir, max_wall_seconds)
    if smrf["ground_smrf"] is None:
        return {
            "min_ground": min_ground, "smrf_ground": None, "fallback_reason": smrf["fallback_reason"],
            "method_note": "SMRF ground did not complete in budget; reporting min-ground residual only.",
        }

    _write_tif(run_dir / "tls_ground_smrf_1m.tif", smrf["ground_smrf"], smrf["grid"]["transform"], smrf["grid"]["crs"])
    clearance = _clearance_mask(footprints, smrf["grid"]["transform"], smrf["ground_smrf"].shape, GROUND_CLEARANCE_M)
    dtm_on_smrf = _resample_to_grid(dtm_path, smrf["grid"]["transform"], smrf["ground_smrf"].shape, smrf["grid"]["crs"])
    valid = (smrf["ground_smrf"] != -9999.0) & clearance & np.isfinite(dtm_on_smrf)
    n = int(valid.sum())
    if n:
        delta = smrf["ground_smrf"][valid].astype("float64") - dtm_on_smrf[valid]
        smrf_ground = {
            "n": n, "median_delta_m": float(np.median(delta)), "p95_abs_delta_m": float(np.percentile(np.abs(delta), 95)),
            "elapsed_s": smrf["elapsed_s"],
        }
    else:
        smrf_ground = {"n": 0, "median_delta_m": None, "p95_abs_delta_m": None, "elapsed_s": smrf["elapsed_s"]}

    return {
        "min_ground": min_ground, "smrf_ground": smrf_ground, "fallback_reason": None,
        "method_note": (
            "PDAL writers.gdal has no per-cell-percentile output_type; SMRF ground classification "
            "(the spec's own offered fallback) substitutes for a literal numpy 5th-percentile-per-cell, "
            "which would require extracting and binning ~262M raw points -- not tractable in this "
            "check's time budget. Reported as SMRF, not relabelled as a percentile."
        ),
    }


# ---------------------------------------------------------------------------
# G1 lite: facade bias report (2.5D surface vs TLS DSM), REPORT ONLY
# ---------------------------------------------------------------------------

def facade_svf(directions, weights, visible, normals):
    """SVF against the same denominator wp04_sites.facade_svf_irradiation uses
    (an unobstructed vertical wall reads ~0.5, not 1.0)."""
    mask = hemisphere_mask(directions, normals)
    dotprod = normals @ directions.T
    cw = weights * directions[:, 2]
    denom = float(cw.sum())
    front_visible = visible & mask
    return (front_visible * weights[None, :] * dotprod).sum(axis=1) / denom


def storey_bin(height_above_ground: np.ndarray) -> np.ndarray:
    idx = np.digitize(height_above_ground, STOREY_BIN_EDGES[1:], right=False)
    idx = np.clip(idx, 0, len(STOREY_BIN_LABELS) - 1)
    return np.array(STOREY_BIN_LABELS, dtype=object)[idx]


def compute_g1_lite(
    data_root: Path, tmp_dir: Path, tls: dict, als_bundle, device: str, rng: np.random.Generator
) -> dict:
    directions, weights = generate_tregenza_patches()
    als_surface, als_transform, als_crs, als_is_building = als_bundle[:4]

    native_dtm, native_fp, _native_roads = resolve_native_paths(SITE_KEY, data_root)
    footprints_gdf = gpd.read_file(native_fp).reset_index(drop=True)
    facade_pts = svf_sampling.sample_facade_points(footprints_gdf, native_dtm)

    obs_xy = np.column_stack([facade_pts["x"].to_numpy(), facade_pts["y"].to_numpy()])
    tls_shape = tls["dsm_1m"].shape
    inv = ~tls["grid_1m"]["transform"]
    cols_tls = np.floor(inv.a * obs_xy[:, 0] + inv.b * obs_xy[:, 1] + inv.c).astype(int)
    rows_tls = np.floor(inv.d * obs_xy[:, 0] + inv.e * obs_xy[:, 1] + inv.f).astype(int)
    in_bounds = (rows_tls >= 0) & (rows_tls < tls_shape[0]) & (cols_tls >= 0) & (cols_tls < tls_shape[1])
    covered = np.zeros(len(obs_xy), dtype=bool)
    covered[in_bounds] = tls["dsm_1m"][rows_tls[in_bounds], cols_tls[in_bounds]] != -9999.0
    if covered.sum() == 0:
        return {"n": 0, "note": "no facade points fell inside the TLS scanned extent"}

    idx_covered = np.where(covered)[0]
    if len(idx_covered) > FACADE_MAX_POINTS:
        idx_covered = rng.choice(idx_covered, size=FACADE_MAX_POINTS, replace=False)
    facade_sub = facade_pts.iloc[idx_covered].reset_index(drop=True)

    obs_xy_sub = np.column_stack([facade_sub["x"].to_numpy(), facade_sub["y"].to_numpy()])
    obs_z_sub = facade_sub["z"].to_numpy(dtype="float64")
    normals_sub = np.column_stack(
        [facade_sub["normal_x"].to_numpy(), facade_sub["normal_y"].to_numpy(), facade_sub["normal_z"].to_numpy()]
    )

    dtm_on_tls = _resample_to_grid(
        data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", tls["grid_1m"]["transform"], tls["dsm_1m"].shape, tls["grid_1m"]["crs"]
    )
    tls_dsm_filled = np.where(tls["dsm_1m"] != -9999.0, tls["dsm_1m"], dtm_on_tls).astype("float32")

    vis_als, _ob_als = patch_visibility(
        als_surface, als_transform, obs_xy_sub, directions=directions, is_building=als_is_building,
        obs_height_m=OBS_HEIGHT_M, obs_z=obs_z_sub, max_dist_m=MAX_DIST_M, step_m=CELL_M, device=device,
    )
    vis_tls, _ob_tls = patch_visibility(
        tls_dsm_filled, tls["grid_1m"]["transform"], obs_xy_sub, directions=directions,
        obs_height_m=OBS_HEIGHT_M, obs_z=obs_z_sub, max_dist_m=MAX_DIST_M, step_m=CELL_M, device=device,
    )
    svf_als = facade_svf(directions, weights, vis_als, normals_sub)
    svf_tls = facade_svf(directions, weights, vis_tls, normals_sub)
    delta = svf_als - svf_tls

    bins = storey_bin(facade_sub["height_above_ground"].to_numpy(dtype="float64"))
    rows = []
    for label in STOREY_BIN_LABELS:
        m = bins == label
        n_bin = int(m.sum())
        if n_bin == 0:
            rows.append({"storey_bin": label, "n": 0})
            continue
        d = delta[m]
        share_pos = float(np.mean(d > 0))
        rows.append({
            "storey_bin": label, "n": n_bin,
            "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)),
            "sign_consistency": max(share_pos, 1.0 - share_pos),
        })

    return {
        "n": len(facade_sub), "n_total_in_extent": int(covered.sum()),
        "storey_bins": rows,
        "overall_mean_delta": float(np.mean(delta)), "overall_median_delta": float(np.median(delta)),
        "note": "REPORT ONLY -- the 2.5D facade layer is NOT ACCEPTED (2026-09-15); this does not change solar_gates status text.",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _footprints_gdf(data_root: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(data_root / f"data/{SITE_KEY}/buildings_extended_700m.gpkg")


def run(data_root: Path, run_dir: Path, device: str, max_wall_seconds: float = MAX_PDAL_WALL_SECONDS, seed: int = 0) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    tls = build_tls_rasters(data_root, tmp_dir, max_wall_seconds)
    _write_tif(run_dir / "tls_dsm_1m.tif", tls["dsm_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"])
    _write_tif(run_dir / "tls_ground_1m.tif", tls["ground_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"])
    if tls["dsm_0p5m"] is not None:
        _write_tif(run_dir / "tls_dsm_0p5m.tif", tls["dsm_0p5m"], tls["grid_0p5m"]["transform"], tls["grid_0p5m"]["crs"])

    footprints = _footprints_gdf(data_root)

    registration = registration_residual(
        tls["ground_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"],
        data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", footprints,
    )
    (run_dir / "registration.json").write_text(json.dumps(registration, indent=1, default=str))

    als_bundle = build_site_surface(SITE_KEY, data_root, CELL_M, tmp_dir)

    g2 = compute_g2(data_root, tls, footprints, als_bundle, device, rng)
    (run_dir / "g2_result.json").write_text(json.dumps(g2, indent=1))

    g1_lite = compute_g1_lite(data_root, tmp_dir, tls, als_bundle, device, rng)
    (run_dir / "g1_lite.json").write_text(json.dumps(g1_lite, indent=1))

    manifest_path = write_run_manifest(
        run_dir, cell_m=CELL_M, obs_height_m=OBS_HEIGHT_M, max_dist_m=MAX_DIST_M, step_m=CELL_M,
        sampling_rule=f"G2 alley-width classes on TLS-covered ground cells; G1-lite <= {FACADE_MAX_POINTS} facade points",
        device=device,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["pdal_version"] = _pdal_version()
    manifest["point_counts"] = tls["point_counts"]
    manifest["pdal_elapsed_s"] = {"1m_pass": tls["elapsed_1m_s"], "total": tls["total_elapsed_s"]}
    manifest["pdal_fallback_reason"] = tls["fallback_reason"]
    manifest["e57_scans"] = tls["scans"]
    manifest["e57_z_range"] = list(E57_Z_RANGE)
    manifest["sky"]["patches"] = int(P1_SKY_PATCHES)
    manifest_path.write_text(json.dumps(manifest, indent=1))

    report_path = run_dir / "report.md"
    report_path.write_text(_render_report(manifest, registration, g2, g1_lite, tls))

    return {"manifest": manifest, "registration": registration, "g2": g2, "g1_lite": g1_lite, "tls": tls}


def _render_report(manifest, registration, g2, g1_lite, tls) -> str:
    utc = manifest["_utc"]
    lines = [f"# WP-03 TLS validity floors -- {utc}", "",
             f"device={manifest['device']}, pdal={manifest.get('pdal_version')}, "
             f"sky_patches={manifest['sky']['patches']}", ""]

    lines += ["## PDAL", "",
              f"e57 scans: {len(tls['scans'])}, point counts: {tls['point_counts']}",
              f"1 m pass: {tls['elapsed_1m_s']:.0f}s, total: {tls['total_elapsed_s']:.0f}s"]
    if tls["fallback_reason"]:
        lines.append(f"0.5 m sensitivity pass SKIPPED: {tls['fallback_reason']}")
    lines.append("")

    lines += ["## Registration residual (TLS ground vs ALS DTM, replaces the missing .rcp RMSE)", "",
              f"n={registration['n']}, clearance>={registration['clearance_m']} m from any footprint",
              f"median delta = {registration['median_delta_m']}", f"p95 |delta| = {registration['p95_abs_delta_m']}",
              f"planar shift = {registration['shift']}", ""]

    lines += ["## G2 -- alley-width validity floor", "",
              f"floor label: **{g2['floor']}**", "",
              "| class | n | r | median delta | p95 |delta| | share<=0.10 |",
              "|---|---|---|---|---|---|"]
    for row in g2["classes"]:
        lines.append(
            f"| {row['class']} | {row['n']} | {row.get('r')} | {row.get('median_delta')} | "
            f"{row.get('p95_abs_delta')} | {row.get('share_within_tol')} |"
        )
    if "sensitivity_0p5m" in g2:
        lines += ["", "### 0.5 m TLS DSM sensitivity", "", "| class | n | median |delta| | p95 |delta| |", "|---|---|---|---|"]
        for row in g2["sensitivity_0p5m"]["classes"]:
            lines.append(f"| {row['class']} | {row['n']} | {row.get('median_abs_delta')} | {row.get('p95_abs_delta')} |")
    lines.append("")

    lines += ["## G1 lite -- facade bias (REPORT ONLY, 2.5D facade layer NOT ACCEPTED)", "",
              f"n={g1_lite.get('n')} of {g1_lite.get('n_total_in_extent')} facade points in the scanned extent",
              f"overall mean delta = {g1_lite.get('overall_mean_delta')}, "
              f"median delta = {g1_lite.get('overall_median_delta')}", "",
              "| storey bin | n | mean delta | median delta | sign consistency |",
              "|---|---|---|---|---|"]
    for row in g1_lite.get("storey_bins", []):
        lines.append(
            f"| {row['storey_bin']} | {row['n']} | {row.get('mean_delta')} | "
            f"{row.get('median_delta')} | {row.get('sign_consistency')} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WP-03B orchestration (docs/wp03b_tls_diagnostic_spec.md)
# ---------------------------------------------------------------------------

def load_tls_rasters(rasters_dir: Path) -> dict:
    """Reload the phase-1 PDAL-extraction outputs (`build_tls_rasters` +
    `_write_tif`) from disk -- WP-03B does not re-run the ~430s PDAL pass; it
    reuses tls_dsm_1m.tif / tls_ground_1m.tif [/ tls_dsm_0p5m.tif] plus the
    sibling `tls_build_meta.json` (point counts, elapsed times) written
    alongside them by whoever last called `build_tls_rasters`."""
    def _read(path: Path):
        with rasterio.open(path) as ds:
            return ds.read(1).astype("float32"), ds.transform, ds.crs

    dsm_1m, transform_1m, crs_1m = _read(rasters_dir / "tls_dsm_1m.tif")
    ground_1m, _t, _c = _read(rasters_dir / "tls_ground_1m.tif")

    dsm_0p5m, grid_0p5m = None, None
    p05 = rasters_dir / "tls_dsm_0p5m.tif"
    if p05.exists():
        dsm_0p5m, transform_05, crs_05 = _read(p05)
        grid_0p5m = {"transform": transform_05, "crs": crs_05, "shape": dsm_0p5m.shape}

    meta_path = rasters_dir / "tls_build_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "dsm_1m": dsm_1m, "ground_1m": ground_1m,
        "grid_1m": {"transform": transform_1m, "crs": crs_1m, "shape": dsm_1m.shape},
        "dsm_0p5m": dsm_0p5m, "grid_0p5m": grid_0p5m,
        "scans": meta.get("scans"), "point_counts": meta.get("point_counts"),
        "elapsed_1m_s": meta.get("elapsed_1m_s"), "total_elapsed_s": meta.get("total_elapsed_s"),
        "fallback_reason": meta.get("fallback_reason"),
    }


def _render_variant_table(variant: dict) -> list[str]:
    lines = [
        f"### variant `{variant['variant']}` -- floor: **{variant['floor']}** "
        f"(n_candidates={variant['n_candidates']}, n_used={variant['n_used']})",
        "",
        "| class | n | r | median delta | p95 |delta| | share<=0.10 |",
        "|---|---|---|---|---|---|",
    ]
    for row in variant["classes"]:
        lines.append(
            f"| {row['class']} | {row['n']} | {row.get('r')} | {row.get('median_delta')} | "
            f"{row.get('p95_abs_delta')} | {row.get('share_within_tol')} |"
        )
    lines.append("")
    return lines


def _render_report_v2(manifest: dict, coverage: dict, variants: dict, shift: dict, ground_def: dict) -> str:
    lines = [f"# WP-03B TLS diagnostic -- {manifest['_utc']}", "",
             f"device={manifest['device']}, sky_patches={manifest['sky']['patches']}", ""]

    lines += ["## Deliverable 1 -- coverage diagnostic", "",
              f"share uncovered of total TLS-grid cells: {coverage['share_uncovered_of_total']}",
              f"of uncovered cells: inside a 2019 footprint = {coverage['share_uncovered_inside_footprint']}, "
              f"on open ground = {coverage['share_uncovered_on_ground']}",
              f"n_uncovered={coverage['n_uncovered']} "
              f"(inside footprint={coverage['n_uncovered_inside_footprint']}, on ground={coverage['n_uncovered_on_ground']})",
              "coverage map: coverage_map.png (withheld class)", ""]

    lines += ["## Deliverable 2 -- G2 surface variants", ""]
    for key in ("a_dtm_fill_merged", "b_als_fill", "c_covered_only_observers", "d_als_fill_shift_corrected"):
        if key in variants:
            lines += _render_variant_table(variants[key])
            if key == "c_covered_only_observers" and variants.get("coverage_share_note"):
                lines += [f"note: {variants['coverage_share_note']}", ""]
    lines.append("")

    lines += ["## Deliverable 3 -- shift check", "",
              f"detected (dx, dy) cells (from this run's `registration_residual`): {shift['detected']}",
              "", "| variant | n | median delta (m) | p95 |delta| (m) |", "|---|---|---|---|"]
    for label, key in [
        ("baseline (unshifted)", "baseline"), ("detected shift", "detected_shift"),
        ("origin +0.5 cell (x)", "origin_plus_0p5_cell_x"), ("origin -0.5 cell (x)", "origin_minus_0p5_cell_x"),
    ]:
        r = shift[key]
        lines.append(f"| {label} | {r['n']} | {r['median_delta_m']} | {r['p95_abs_delta_m']} |")
    lines += ["", f"closest to zero median: **{shift['closest_to_zero']}**", ""]

    lines += ["## Deliverable 4 -- ground definition check (min vs SMRF)", ""]
    md = ground_def["min_ground"]
    lines.append(f"min ground: n={md['n']}, median delta={md['median_delta_m']} m, p95={md['p95_abs_delta_m']} m")
    if ground_def.get("smrf_ground"):
        sd = ground_def["smrf_ground"]
        lines.append(
            f"SMRF ground: n={sd['n']}, median delta={sd['median_delta_m']} m, p95={sd['p95_abs_delta_m']} m "
            f"(elapsed {sd.get('elapsed_s', 0):.0f}s)"
        )
    else:
        lines.append(f"SMRF ground: SKIPPED -- {ground_def.get('fallback_reason')}")
    lines.append(f"note: {ground_def.get('method_note')}")
    lines.append("")
    return "\n".join(lines)


def run_wp03b(data_root: Path, run_dir: Path, rasters_dir: Path, device: str, seed: int = 0) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tls = load_tls_rasters(rasters_dir)
    footprints = _footprints_gdf(data_root)
    als_bundle = build_site_surface(SITE_KEY, data_root, CELL_M, tmp_dir)
    als_surface, als_transform, als_crs = als_bundle[0], als_bundle[1], als_bundle[2]

    coverage, building_mask = coverage_diagnostic(
        tls["dsm_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"], footprints
    )
    (run_dir / "coverage_diagnostic.json").write_text(json.dumps(coverage, indent=1))
    write_coverage_map_png(run_dir / "coverage_map.png", tls["dsm_1m"], building_mask)

    variants = compute_g2_variants(data_root, tls, footprints, als_bundle, device)

    registration = registration_residual(
        tls["ground_1m"], tls["grid_1m"]["transform"], tls["grid_1m"]["crs"],
        data_root / f"data/{SITE_KEY}/dtm_extended_700m.tif", footprints,
    )
    shift = shift_check(tls, footprints, data_root, registration)

    shifted_transform = _shift_transform(
        tls["grid_1m"]["transform"], shift["detected"]["dx_cells"], shift["detected"]["dy_cells"]
    )
    filled_d = als_fill_surface(tls["dsm_1m"], shifted_transform, tls["grid_1m"]["crs"], als_surface, als_transform, als_crs)
    variants["d_als_fill_shift_corrected"] = compute_g2_variant(
        footprints, als_bundle,
        tls["dsm_1m"], shifted_transform, tls["grid_1m"]["crs"],
        filled_d, shifted_transform, device, "d_als_fill_shift_corrected",
    )
    (run_dir / "g2_result_v2.json").write_text(json.dumps({"variants": variants}, indent=1))
    (run_dir / "shift_check.json").write_text(json.dumps(shift, indent=1))

    ground_def = ground_definition_check(data_root, tmp_dir, tls, footprints, run_dir)
    (run_dir / "ground_definition_check.json").write_text(json.dumps(ground_def, indent=1))

    manifest = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": device, "seed": seed,
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "rasters_dir": str(rasters_dir),
        "pdal_version": _pdal_version(),
        "point_counts": tls.get("point_counts"),
        "spec": "docs/wp03b_tls_diagnostic_spec.md",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (run_dir / "report_v2.md").write_text(_render_report_v2(manifest, coverage, variants, shift, ground_def))

    return {"manifest": manifest, "coverage": coverage, "variants": variants, "shift": shift, "ground_definition": ground_def}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-wall-seconds", type=float, default=MAX_PDAL_WALL_SECONDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wp03b", action="store_true", help="run the WP-03B diagnostic (deliverables 1-5) instead of phase 1")
    ap.add_argument("--rasters-dir", type=Path, default=None, help="--wp03b: dir holding tls_dsm_1m.tif etc (skips the PDAL pass)")
    args = ap.parse_args()

    run_dir = args.run_dir
    device = args.device or default_device()

    if args.wp03b:
        if run_dir is None:
            run_id = "wp03_tls_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = REPO_ROOT / "runs" / run_id
        rasters_dir = args.rasters_dir or run_dir
        result = run_wp03b(args.data_root, run_dir, rasters_dir, device, args.seed)
        print(json.dumps({"run_dir": str(run_dir), "coverage": result["coverage"]}, indent=1))
        return 0

    if run_dir is None:
        run_id = "wp03_tls_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = REPO_ROOT / "runs" / run_id

    result = run(args.data_root, run_dir, device, args.max_wall_seconds, args.seed)
    print(json.dumps({"run_dir": str(run_dir), "floor": result["g2"]["floor"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WP-05: citywide PILOT — sampling frame, 9-stratum draw, tiled evaluation.

Spec: docs/wp05_pilot_spec.md. Produces the four go/no-go facts (timing and
memory extrapolation to 3M cells, resolution sensitivity, stratum coverage,
favela share) from a 1% stratified pilot of the DTM's own 5 m grid. Never
claims the full 3M run — that is a later, separate task gated on this one.

The obstruction surface (wp02_surface.build_surface) and the per-patch
visibility engine (wp02_horizon.patch_visibility) are accepted and are called
here unmodified; this module only adds the sampling frame, the tile/halo
loop that clips city-wide inputs down to something build_surface can handle
at 1 m, and the pilot bookkeeping (strata, favela flag, checkpointed
Parquet, timing).

Run: python -m src.brisa_solar.wp05_pilot --cell-sizes 5,2,1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from scipy import ndimage

from . import wp02_sky
from .constants import P1_SKY_PATCHES, REPO_ROOT, load_params
from .wp02_horizon import patch_visibility, write_run_manifest
from .wp02_surface import build_surface, load_surface
from src.svf_v2.compute import generate_tregenza_patches

#: Stratification bin edges. PILOT DEFAULTS, not decisions — spec "Numbers you
#: may not type": these get written into the run manifest, never into
#: config/params.yaml, until the PI signs off post-pilot.
SLOPE_BINS_DEG = (5.0, 15.0)
DENSITY_BINS = (0.10, 0.25, 0.45)
N_STRATA = 9

TILE_M = 2000.0
HALO_M = 500.0

#: Spec text, not a params.yaml key (params.yaml's sampling.stratum_floor=50000
#: is the FULL 3M-run floor, a different number from the pilot's own floor).
PILOT_FRACTION = 0.01
PILOT_STRATUM_FLOOR = 1000


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Sampling frame
# ---------------------------------------------------------------------------

def compute_slope_deg(dtm: np.ndarray, cell_m: float) -> np.ndarray:
    """Slope in degrees via np.gradient. NaN propagates from DTM nodata neighbours."""
    gy, gx = np.gradient(dtm, cell_m, cell_m)
    return np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))


def build_frame(
    dtm: np.ndarray,
    is_building: np.ndarray,
    *,
    cell_m: float,
    fabric_coverage_threshold: float,
    fabric_footprint_distance_m: float,
) -> dict:
    """Sampling frame over the DTM's own grid.

    Rules applied in spec order, each removal recorded against the mask
    surviving the previous rule (spec: "Record how many cells each rule
    removes"):
      1. not a building cell
      2. within the municipality (DTM valid / finite)
      3. fabric_coverage >= threshold (100 m x 100 m building-cell fraction)
         AND within fabric_footprint_distance_m of a footprint
    """
    total = int(dtm.size)
    window_cells = max(1, int(round(100.0 / cell_m)))
    coverage = ndimage.uniform_filter(
        is_building.astype("float32"), size=window_cells, mode="constant"
    )
    # distance_transform_edt gives, for each cell, the distance (in cells) to
    # the nearest True cell in its input; feeding ~is_building gives distance
    # to the nearest BUILDING cell, which is what "within X m of a footprint"
    # means. A building cell itself gets distance 0.
    dist_m = ndimage.distance_transform_edt(~is_building) * cell_m

    keep = ~is_building
    removed_building = int(total - keep.sum())

    dtm_valid = np.isfinite(dtm)
    before = keep.copy()
    keep &= dtm_valid
    removed_invalid_dtm = int((before & ~keep).sum())

    before = keep.copy()
    keep &= coverage >= fabric_coverage_threshold
    removed_low_coverage = int((before & ~keep).sum())

    before = keep.copy()
    keep &= dist_m <= fabric_footprint_distance_m
    removed_far_from_footprint = int((before & ~keep).sum())

    return {
        "in_frame": keep,
        "coverage": coverage,
        "dist_m": dist_m,
        "total_cells": total,
        "frame_cells": int(keep.sum()),
        "removal_counts": {
            "rule_order": [
                "not_building_cell",
                "within_municipality_dtm_valid",
                "fabric_coverage_ge_threshold",
                "within_fabric_footprint_distance_m",
            ],
            "removed_building_cells": removed_building,
            "removed_invalid_dtm_cells": removed_invalid_dtm,
            "removed_low_fabric_coverage_cells": removed_low_coverage,
            "removed_far_from_footprint_cells": removed_far_from_footprint,
        },
        "fabric_coverage_threshold": fabric_coverage_threshold,
        "fabric_footprint_distance_m": fabric_footprint_distance_m,
        "fabric_window_cells": window_cells,
    }


def assign_strata(
    slope_deg: np.ndarray,
    coverage: np.ndarray,
    *,
    slope_bins=SLOPE_BINS_DEG,
    density_bins=DENSITY_BINS,
) -> np.ndarray:
    """9 strata = 3 slope bins x 3 density bins, stratum = slope_bin*3 + density_bin."""
    slope_bin = np.digitize(slope_deg, slope_bins)          # 0: <5, 1: 5-15, 2: >=15
    density_bin = np.digitize(coverage, density_bins[1:])   # bin edges (0.25, 0.45) -> 0/1/2
    return (slope_bin * 3 + density_bin).astype("int16")


def stratified_pilot(
    frame_mask: np.ndarray,
    stratum: np.ndarray,
    *,
    pilot_fraction: float,
    floor: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Draw a stratified pilot. Deterministic ordering before any random draw
    (lexsort on row/col) so a fixed seed reproduces identical cell ids.
    """
    rows, cols = np.where(frame_mask)
    strata_vals = stratum[rows, cols]
    order = np.lexsort((cols, rows))
    rows, cols, strata_vals = rows[order], cols[order], strata_vals[order]

    rng = np.random.default_rng(seed)
    picked_chunks = []
    per_stratum = {}
    for s in range(N_STRATA):
        idx_s = np.where(strata_vals == s)[0]
        n_s = int(len(idx_s))
        if n_s == 0:
            per_stratum[s] = {"frame_n": 0, "pilot_n": 0, "target_n": 0}
            continue
        target = max(int(round(pilot_fraction * n_s)), floor)
        target = min(target, n_s)
        picked = rng.choice(idx_s, size=target, replace=False)
        picked.sort()
        picked_chunks.append(picked)
        per_stratum[s] = {"frame_n": n_s, "pilot_n": int(target), "target_n": int(target)}

    chosen = np.concatenate(picked_chunks) if picked_chunks else np.array([], dtype=int)
    return rows[chosen], cols[chosen], strata_vals[chosen], per_stratum


# ---------------------------------------------------------------------------
# Tile/halo window assembly (pure array op — tested independently of file I/O)
# ---------------------------------------------------------------------------

def crop_surface_to_bounds(
    surface: np.ndarray,
    transform: Affine,
    bounds: tuple[float, float, float, float],
    is_building: np.ndarray | None = None,
):
    """Crop `surface` (and optional `is_building`) to world `bounds`, clamped
    to the array's own extent, and return the adjusted transform.

    This is the halo-windowing primitive: cropping to tile+halo and shifting
    the transform must leave visibility for an interior observer unchanged
    versus evaluating on the full surface, provided the halo covers
    max_dist_m (tested in tests/test_wp05_pilot.py).
    """
    minx, miny, maxx, maxy = bounds
    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    row0 = max(0, int(np.floor(win.row_off)))
    col0 = max(0, int(np.floor(win.col_off)))
    row1 = min(surface.shape[0], int(np.ceil(win.row_off + win.height)))
    col1 = min(surface.shape[1], int(np.ceil(win.col_off + win.width)))
    cropped = surface[row0:row1, col0:col1]
    cropped_ib = is_building[row0:row1, col0:col1] if is_building is not None else None
    new_transform = transform * Affine.translation(col0, row0)
    return cropped, new_transform, cropped_ib


def tile_index_for_xy(x: np.ndarray, y: np.ndarray, origin_x: float, origin_y: float, tile_m: float = TILE_M):
    i = np.floor((x - origin_x) / tile_m).astype("int64")
    j = np.floor((y - origin_y) / tile_m).astype("int64")
    return i, j


def tile_bounds(i: int, j: int, origin_x: float, origin_y: float, tile_m: float = TILE_M):
    return (origin_x + i * tile_m, origin_y + j * tile_m, origin_x + (i + 1) * tile_m, origin_y + (j + 1) * tile_m)


def halo_bounds(b: tuple[float, float, float, float], halo_m: float = HALO_M):
    minx, miny, maxx, maxy = b
    return (minx - halo_m, miny - halo_m, maxx + halo_m, maxy + halo_m)


# ---------------------------------------------------------------------------
# Visibility packing (np.packbits/unpackbits round-trip)
# ---------------------------------------------------------------------------

def pack_visibility(vis: np.ndarray) -> np.ndarray:
    """(n, P) bool -> (n, ceil(P/8)) uint8, one row per observer."""
    return np.packbits(vis, axis=-1)


def unpack_visibility(packed: np.ndarray, n_patches: int) -> np.ndarray:
    """Inverse of pack_visibility. Trims the padding packbits adds to a byte boundary."""
    bits = np.unpackbits(packed, axis=-1)
    return bits[:, :n_patches].astype(bool)


# ---------------------------------------------------------------------------
# One tile: clip DTM + footprints to tile+halo, build_surface, evaluate
# ---------------------------------------------------------------------------

@dataclass
class TileTiming:
    tile_id: str
    n_obs: int
    build_s: float
    engine_s: float
    peak_gb: float


def _write_clipped_dtm(dtm_dataset, halo_b, out_tif: Path) -> Path:
    win = rasterio.windows.from_bounds(*halo_b, transform=dtm_dataset.transform)
    win = win.round_lengths(op="ceil").round_offsets(op="floor")
    arr = dtm_dataset.read(1, window=win, boundless=True, fill_value=dtm_dataset.nodata)
    transform = dtm_dataset.window_transform(win)
    profile = dict(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs=dtm_dataset.crs, transform=transform, nodata=dtm_dataset.nodata,
    )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)
    return out_tif


def _write_clipped_footprints(footprints_path: Path, halo_b, out_gpkg: Path) -> tuple[Path, int]:
    gdf = gpd.read_file(footprints_path, bbox=halo_b)
    if len(gdf) == 0:
        gdf = gpd.read_file(footprints_path, rows=0)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    return out_gpkg, int(len(gdf))


def evaluate_tile(
    *,
    dtm_dataset,
    footprints_path: Path,
    tile_i: int,
    tile_j: int,
    origin_x: float,
    origin_y: float,
    cell_m: float,
    obs_df: pd.DataFrame,
    directions: np.ndarray,
    sky: "wp02_sky.CumulativeSky",
    device: str,
    tmp_dir: Path,
    tile_m: float = TILE_M,
    halo_m: float = HALO_M,
) -> tuple[pd.DataFrame, TileTiming]:
    tb = tile_bounds(tile_i, tile_j, origin_x, origin_y, tile_m)
    hb = halo_bounds(tb, halo_m)
    tile_id = f"{tile_i}_{tile_j}"

    t0 = time.perf_counter()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tile_dtm = _write_clipped_dtm(dtm_dataset, hb, tmp_dir / f"tile_{tile_id}_dtm.tif")
    tile_fps, n_fp = _write_clipped_footprints(footprints_path, hb, tmp_dir / f"tile_{tile_id}_fp.gpkg")
    out_stem = tmp_dir / f"tile_{tile_id}"
    surface_tif = build_surface(tile_dtm, tile_fps, cell_m, out_stem)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    meta_json = out_stem.with_name(out_stem.stem + "_meta.json")
    surface, transform, _crs, is_building = load_surface(surface_tif, is_building_tif)
    build_s = time.perf_counter() - t0

    obs_xy = obs_df[["x", "y"]].to_numpy(dtype="float64")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    vis, on_building = patch_visibility(
        surface, transform, obs_xy,
        directions=directions, is_building=is_building,
        max_dist_m=halo_m, march_sampling="nearest", device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_gb = float("nan")
    engine_s = time.perf_counter() - t1

    svf = sky.svf(vis.astype(float))
    irr = sky.irradiation(vis.astype(float))
    packed = pack_visibility(vis)

    out = obs_df.copy()
    out["cell_m"] = cell_m
    out["tile_id"] = tile_id
    out["n_footprints_in_tile_halo"] = n_fp
    out["on_building"] = on_building
    out["svf"] = svf
    out["irradiation_kwh_m2"] = irr
    out["visibility_packed"] = [row.tobytes() for row in packed]

    for f in (tile_dtm, tile_fps, surface_tif, is_building_tif, meta_json):
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass

    timing = TileTiming(tile_id=tile_id, n_obs=int(len(obs_df)), build_s=build_s, engine_s=engine_s, peak_gb=peak_gb)
    return out, timing


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_frame_and_pilot(run_dir: Path, *, dtm_path: Path, footprints_path: Path, favelas_path: Path) -> pd.DataFrame:
    """Build the sampling frame + 9-stratum pilot draw. Writes diagnostics + pilot_cells.parquet."""
    params = load_params()
    domain = params["domain"]
    sampling = params["sampling"]

    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        dtm = np.where(np.isclose(dtm, nodata, rtol=1e-3), np.nan, dtm)
    cell_m = abs(transform.a)

    surface_stem = run_dir / "artifacts" / "frame_5m"
    surface_tif = build_surface(dtm_path, footprints_path, cell_m, surface_stem)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    _surface5m, _t5m, _crs5m, is_building = load_surface(surface_tif, is_building_tif)

    frame = build_frame(
        dtm, is_building,
        cell_m=cell_m,
        fabric_coverage_threshold=domain["fabric_coverage_threshold"],
        fabric_footprint_distance_m=domain["fabric_footprint_distance_m"],
    )
    slope_deg = compute_slope_deg(dtm, cell_m)
    stratum = assign_strata(slope_deg, frame["coverage"])

    rows, cols, strata_vals, per_stratum = stratified_pilot(
        frame["in_frame"], stratum,
        pilot_fraction=PILOT_FRACTION,
        floor=PILOT_STRATUM_FLOOR,
        seed=sampling["random_seed"],
    )

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    favelas = gpd.read_file(favelas_path)
    if favelas.crs is not None and str(favelas.crs) != str(crs):
        favelas = favelas.to_crs(crs)
    pts = gpd.GeoDataFrame(
        {"cell_id": np.arange(len(xs))},
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs,
    )
    joined = gpd.sjoin(pts, favelas[["geometry"]], how="left", predicate="within")
    favela_flag = ~joined.groupby("cell_id")["index_right"].first().isna()
    favela_flag = favela_flag.reindex(np.arange(len(xs)), fill_value=False).to_numpy()

    pilot_df = pd.DataFrame({
        "cell_id": [f"r{r}c{c}" for r, c in zip(rows, cols)],
        "row": rows,
        "col": cols,
        "x": xs,
        "y": ys,
        "stratum": strata_vals,
        "slope_deg": slope_deg[rows, cols],
        "fabric_coverage": frame["coverage"][rows, cols],
        "favela": favela_flag,
    })

    run_dir.mkdir(parents=True, exist_ok=True)
    pilot_df.to_parquet(run_dir / "pilot_cells.parquet", index=False)

    diagnostics = {
        "_utc": _utc_now(),
        "dtm_path": str(dtm_path),
        "footprints_path": str(footprints_path),
        "favelas_path": str(favelas_path),
        "grid_cell_m": cell_m,
        "dtm_shape": list(dtm.shape),
        "dtm_bounds": list(bounds),
        "frame": {
            "total_cells": frame["total_cells"],
            "frame_cells": frame["frame_cells"],
            "removal_counts": frame["removal_counts"],
            "fabric_coverage_threshold": frame["fabric_coverage_threshold"],
            "fabric_footprint_distance_m": frame["fabric_footprint_distance_m"],
        },
        "strata": {
            "slope_bins_deg_PILOT_DEFAULT": list(SLOPE_BINS_DEG),
            "density_bins_PILOT_DEFAULT": list(DENSITY_BINS),
            "per_stratum": {str(k): v for k, v in per_stratum.items()},
        },
        "pilot": {
            "pilot_fraction": PILOT_FRACTION,
            "pilot_stratum_floor": PILOT_STRATUM_FLOOR,
            "random_seed": sampling["random_seed"],
            "n_pilot_cells": int(len(pilot_df)),
            "n_favela_cells": int(favela_flag.sum()),
            "favela_share": float(favela_flag.mean()) if len(favela_flag) else float("nan"),
        },
        "git_sha": _git_sha(),
    }
    (run_dir / "frame_and_pilot_diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
    return pilot_df


def run_cellsize_pass(
    cell_m: float,
    pilot_df: pd.DataFrame,
    run_dir: Path,
    *,
    dtm_path: Path,
    footprints_path: Path,
    sky: "wp02_sky.CumulativeSky",
    directions: np.ndarray,
    device: str,
    origin_x: float,
    origin_y: float,
    time_budget_s: float | None = None,
) -> dict:
    """Evaluate every pilot cell at one resolution, tiled, checkpointed per tile.

    Resumable: a tile whose output parquet already exists is skipped. Logs
    per-tile timing to <run_dir>/timing_<cell_m>m.jsonl (appended, so a
    resumed run keeps prior tiles' timing).
    """
    tile_dir = run_dir / f"cellsize_{cell_m:g}m" / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / f"cellsize_{cell_m:g}m" / "_tmp"
    timing_path = run_dir / f"timing_{cell_m:g}m.jsonl"

    i, j = tile_index_for_xy(pilot_df["x"].to_numpy(), pilot_df["y"].to_numpy(), origin_x, origin_y)
    pilot_df = pilot_df.assign(_tile_i=i, _tile_j=j)
    tiles = sorted(pilot_df.groupby(["_tile_i", "_tile_j"]).groups.keys())

    started = time.perf_counter()
    stopped_early = False
    with rasterio.open(dtm_path) as dtm_dataset:
        for tile_i, tile_j in tiles:
            tile_id = f"{tile_i}_{tile_j}"
            out_path = tile_dir / f"tile_{tile_id}.parquet"
            if out_path.exists():
                continue
            if time_budget_s is not None and (time.perf_counter() - started) > time_budget_s:
                stopped_early = True
                break
            obs = pilot_df[(pilot_df["_tile_i"] == tile_i) & (pilot_df["_tile_j"] == tile_j)]
            obs = obs.drop(columns=["_tile_i", "_tile_j"])
            out, timing = evaluate_tile(
                dtm_dataset=dtm_dataset, footprints_path=footprints_path,
                tile_i=tile_i, tile_j=tile_j, origin_x=origin_x, origin_y=origin_y,
                cell_m=cell_m, obs_df=obs, directions=directions, sky=sky,
                device=device, tmp_dir=tmp_dir,
            )
            out.to_parquet(out_path, index=False)
            with timing_path.open("a") as fh:
                fh.write(json.dumps({
                    "_utc": _utc_now(), "cell_m": cell_m, **timing.__dict__,
                }) + "\n")
            n_done_so_far = sum(1 for _ in tile_dir.glob("tile_*.parquet"))
            print(
                f"[{cell_m:g}m] tile {tile_id} done: n_obs={timing.n_obs} "
                f"build_s={timing.build_s:.1f} engine_s={timing.engine_s:.1f} "
                f"peak_gb={timing.peak_gb:.3f} ({n_done_so_far}/{len(tiles)} tiles)",
                flush=True,
            )

    tile_files = sorted(tile_dir.glob("tile_*.parquet"))
    n_done_tiles = len(tile_files)
    n_total_tiles = len(tiles)
    if tile_files:
        merged = pd.concat([pd.read_parquet(f) for f in tile_files], ignore_index=True)
        merged.to_parquet(run_dir / f"cell_{cell_m:g}m.parquet", index=False)
    return {
        "cell_m": cell_m,
        "n_tiles_total": n_total_tiles,
        "n_tiles_done": n_done_tiles,
        "stopped_early": stopped_early,
        "wall_s": time.perf_counter() - started,
    }


def compute_extrapolation(run_dir: Path, cell_sizes: list[float], city_sample_n: int) -> dict:
    result = {}
    for cell_m in cell_sizes:
        timing_path = run_dir / f"timing_{cell_m:g}m.jsonl"
        if not timing_path.exists():
            continue
        rows = [json.loads(l) for l in timing_path.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        n_obs = sum(r["n_obs"] for r in rows)
        total_s = sum(r["build_s"] + r["engine_s"] for r in rows)
        peaks = [r["peak_gb"] for r in rows if r["peak_gb"] == r["peak_gb"]]  # drop NaN (cpu device)
        cells_per_s = n_obs / total_s if total_s > 0 else float("nan")
        s_per_1000 = 1000.0 / cells_per_s if cells_per_s > 0 else float("nan")
        peak_gb = max(peaks) if peaks else float("nan")
        proj_hours = (city_sample_n / cells_per_s) / 3600.0 if cells_per_s > 0 else float("nan")
        result[f"{cell_m:g}m"] = {
            "n_measured_cells": n_obs,
            "n_tiles_measured": len(rows),
            "measured_wall_s": total_s,
            "cells_per_s": cells_per_s,
            "s_per_1000_cells": s_per_1000,
            "peak_gb_measured": peak_gb,
            "projected_hours_for_city_sample_n": proj_hours,
            "projected_peak_gb_for_city_sample_n": peak_gb,
            "peak_gb_note": (
                "peak GB is bounded by tile surface size + engine chunk size, not by "
                "total observer count (the engine processes chunk-of-4096 at a time and "
                "tiles are processed sequentially) — so it does not scale with city_sample_n; "
                "reported as the measured peak, not an extrapolation."
            ),
        }
    result["city_sample_n"] = city_sample_n
    result["_utc"] = _utc_now()
    return result


def compute_resolution_sensitivity(run_dir: Path, cell_sizes: list[float]) -> dict:
    frames = {}
    for cell_m in cell_sizes:
        p = run_dir / f"cell_{cell_m:g}m.parquet"
        if p.exists():
            frames[cell_m] = pd.read_parquet(p, columns=["cell_id", "svf", "irradiation_kwh_m2"])
    available = sorted(frames.keys())
    result = {"_utc": _utc_now(), "available_cell_sizes": available}
    for a_idx in range(len(available)):
        for b_idx in range(a_idx + 1, len(available)):
            a, b = available[a_idx], available[b_idx]
            merged = frames[a].merge(frames[b], on="cell_id", suffixes=(f"_{a:g}m", f"_{b:g}m"))
            key = f"{a:g}m_vs_{b:g}m"
            pair = {}
            for metric in ("svf", "irradiation_kwh_m2"):
                va = merged[f"{metric}_{a:g}m"].to_numpy()
                vb = merged[f"{metric}_{b:g}m"].to_numpy()
                diff = va - vb
                pair[metric] = {
                    "n": int(len(diff)),
                    "median_abs_diff": float(np.median(np.abs(diff))),
                    "p95_abs_diff": float(np.percentile(np.abs(diff), 95)),
                    "signed_median_diff": float(np.median(diff)),
                }
            result[key] = pair
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-sizes", default="5,2,1", help="comma-separated cell sizes in metres, in run order")
    ap.add_argument("--run-dir", default=None, help="defaults to a fresh runs/wp05_pilot_<UTC>/ in the main checkout")
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela", help="main checkout, where data/ lives")
    ap.add_argument("--time-budget-hours", type=float, default=None, help="stop a cell-size pass after this many hours of tile work")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    params = load_params()
    cell_sizes = [float(x) for x in args.cell_sizes.split(",")]

    run_dir = Path(args.run_dir) if args.run_dir else data_root / "runs" / ("wp05_pilot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = data_root / params["terrain"]["dtm_city"]
    # params.yaml's canonical_layer value is an unquoted YAML scalar that carries
    # trailing prose after the path ("... — USE THIS, not the raw .shp. ..."); the
    # path itself has no whitespace, so the first token is the real path.
    footprints_path = data_root / params["footprints"]["canonical_layer"].split()[0]
    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"
    epw_path = data_root / params["weather"]["primary_epw"]

    pilot_parquet = run_dir / "pilot_cells.parquet"
    if pilot_parquet.exists():
        pilot_df = pd.read_parquet(pilot_parquet)
    else:
        pilot_df = build_frame_and_pilot(run_dir, dtm_path=dtm_path, footprints_path=footprints_path, favelas_path=favelas_path)

    with rasterio.open(dtm_path) as src:
        origin_x, origin_y = src.bounds.left, src.bounds.bottom

    directions, _weights = generate_tregenza_patches()
    sky = wp02_sky.build(epw_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_budget_s = args.time_budget_hours * 3600.0 if args.time_budget_hours else None

    pass_reports = []
    for cell_m in cell_sizes:
        report = run_cellsize_pass(
            cell_m, pilot_df, run_dir,
            dtm_path=dtm_path, footprints_path=footprints_path,
            sky=sky, directions=directions, device=device,
            origin_x=origin_x, origin_y=origin_y, time_budget_s=time_budget_s,
        )
        pass_reports.append(report)
        # Per-pass manifest (own subdirectory: write_run_manifest's schema is one
        # cell_m per file, and this WP-05 run covers several).
        write_run_manifest(
            run_dir / f"cellsize_{cell_m:g}m", cell_m=cell_m, obs_height_m=1.5, max_dist_m=HALO_M, step_m=cell_m,
            sampling_rule=f"WP-05 pilot: {len(pilot_df)} stratified cells, seed={params['sampling']['random_seed']}",
            device=device,
        )

    # One canonical run_dir/manifest.json (the shape test_all_run_manifests_used_
    # the_same_sky scans for, runs/*/manifest.json — exactly one level deep) so
    # this run participates in the citywide one-sky-resolution check even though
    # it spans several cell sizes, which write_run_manifest's single-cell_m
    # schema cannot represent in one call.
    sky_section = json.dumps(params["sky"], sort_keys=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "_utc": _utc_now(),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_sizes_m": cell_sizes,
        "obs_height_m": 1.5,
        "max_dist_m": HALO_M,
        "sampling_rule": f"WP-05 pilot: {len(pilot_df)} stratified cells, seed={params['sampling']['random_seed']}",
        "device": device,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "params_sky_section_sha256": hashlib.sha256(sky_section.encode()).hexdigest()[:16],
    }, indent=1))

    extrapolation = compute_extrapolation(run_dir, cell_sizes, params["sampling"]["city_sample_n"])
    (run_dir / "extrapolation.json").write_text(json.dumps(extrapolation, indent=1))

    sensitivity = compute_resolution_sensitivity(run_dir, cell_sizes)
    (run_dir / "resolution_sensitivity.json").write_text(json.dumps(sensitivity, indent=1))

    print(json.dumps({"run_dir": str(run_dir), "passes": pass_reports, "extrapolation": extrapolation}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

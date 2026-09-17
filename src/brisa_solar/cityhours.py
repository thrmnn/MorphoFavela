"""CITYHOURS — citywide direct-sun-hour layer at site fidelity.

Spec: docs/cityhours_spec.md. Closes the gap between the per-site layer
(which already expresses the Athens Charter (1943) Point 26 2 h floor via
`wp04_sites.direct_sun_hours`) and the citywide layer (WP-05 FULL, which
only ever stored `svf`/`kwh_m2` + the binary `visibility_packed` mask).

Route (b), the only route this module *publishes*: re-run the horizon
engine over the SAME exhaustive frame WP-05 FULL used
(`runs/wp05_full_20260914T215419Z/frame_cells.parquet` — deterministic
given the domain's decided parameters, so reusing it rather than
rebuilding the raster frame changes nothing and saves the rebuild cost),
emitting continuous per-azimuth horizon angles
(`wp02_horizon.patch_visibility(..., return_horizon=True)`), then the
SAME `wp04_sites.direct_sun_hours` the site numbers use.

Route (a) — sun-hours recovered from WP-05 FULL's already-stored binary
`visibility_packed` mask — is computed ONLY as a lossiness cross-check
against route (b) (`route_a_vs_b_lossiness`). It is never written to
summary.json and never enters the ledger (spec: "never a published
number").

Per-cell output stays in runs/cityhours_<UTC>/ — red line L1, withheld.
Only summary.json (citywide shares/percentiles + each favela's percentile
position in the citywide sun-hour distribution) is publishable-candidate.

Run: python -m src.brisa_solar.cityhours --mode pilot
     python -m src.brisa_solar.cityhours --mode full
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch

from . import wp02_sky
from .constants import P1_SKY_PATCHES, load_params
from .wp02_horizon import patch_visibility
from .wp02_surface import build_surface, load_surface
from .wp04_sites import direct_sun_hours, epw_meta, patch_azimuth_deg
from .wp05_full import (
    PROVISIONAL_STATUS,
    QUANTILES,
    STUDY_FAVELAS,
    SKY_MODEL_LABEL,
    consolidate_tiles,
    favela_summary,
    match_favela_group,
    quantile_block,
)
from .wp05_pilot import (
    HALO_M,
    TILE_M,
    _git_sha,
    _utc_now,
    _write_clipped_dtm,
    _write_clipped_footprints,
    halo_bounds,
    tile_bounds,
    tile_index_for_xy,
    unpack_visibility,
)
from src.svf_v2.compute import generate_tregenza_patches

CELL_M = 1.0

#: The WP-05 FULL run of record — its frame_cells.parquet IS this module's
#: frame (spec "Explicitly OUT": changing the domain thresholds/cell size/seed
#: invalidates the reproduction check, so the frame is reused verbatim rather
#: than rebuilt from the raster rules a second time) and its wp05_full.parquet
#: is route (a)'s only input (the stored visibility_packed mask).
WP05_FULL_RUN_OF_RECORD = "wp05_full_20260914T215419Z"

PILOT_TARGET_FRACTION = 0.015  # midpoint of the project's 1-2% pilot rule


@dataclass
class CityHoursTileTiming:
    tile_id: str
    n_obs: int
    build_s: float
    engine_s: float
    hours_s: float
    peak_gb: float


# ---------------------------------------------------------------------------
# Per-tile evaluation: build_surface + continuous-horizon engine + sun-hours
# ---------------------------------------------------------------------------

def evaluate_tile_hours(
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
    meta: dict,
    reference_days: dict,
    duration_thresholds_h: list[int],
    patch_az_deg: np.ndarray,
    tile_m: float = TILE_M,
    halo_m: float = HALO_M,
) -> tuple[pd.DataFrame, CityHoursTileTiming]:
    """One tile, every frame cell in it: clip DTM+footprints to tile+halo,
    build_surface (unmodified), then patch_visibility with
    return_horizon=True (unmodified engine call, just the flag WP-04 sites
    already use) and direct_sun_hours on both reference days (the SAME
    routine, same duration thresholds). horizon_deg is never written to
    disk — it is (n, 145) float16 per tile, reduced to sun-hours columns
    and discarded immediately, same memory discipline as WP-04's
    evaluate_and_write_parquet.
    """
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
    vis, on_building, horizon_deg = patch_visibility(
        surface, transform, obs_xy,
        directions=directions, is_building=is_building,
        max_dist_m=halo_m, march_sampling="nearest", device=device,
        return_horizon=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_gb = float("nan")
    engine_s = time.perf_counter() - t1

    svf = sky.svf(vis.astype(float))
    irr = sky.irradiation(vis.astype(float))

    t2 = time.perf_counter()
    out = obs_df.copy()
    out["cell_m"] = cell_m
    out["tile"] = tile_id
    out["n_footprints_in_tile_halo"] = n_fp
    out["on_building"] = on_building
    out["svf"] = svf
    out["kwh_m2"] = irr
    out["sky_model"] = SKY_MODEL_LABEL
    for label, date_str in reference_days.items():
        r = direct_sun_hours(horizon_deg, patch_az_deg, date_str, meta, duration_thresholds_h)
        out[f"hours_{label}"] = r["hours_fractional"]
        for k in duration_thresholds_h:
            out[f"ge_{k}h_{label}"] = r[f"ge_{k}h"]
    hours_s = time.perf_counter() - t2

    del horizon_deg, vis

    for f in (tile_dtm, tile_fps, surface_tif, is_building_tif, meta_json):
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass

    timing = CityHoursTileTiming(
        tile_id=tile_id, n_obs=int(len(obs_df)), build_s=build_s,
        engine_s=engine_s, hours_s=hours_s, peak_gb=peak_gb,
    )
    return out, timing


# ---------------------------------------------------------------------------
# Frame (reused verbatim from the WP-05 FULL run of record) + tiling
# ---------------------------------------------------------------------------

def load_frame(data_root: Path) -> pd.DataFrame:
    frame_path = data_root / "runs" / WP05_FULL_RUN_OF_RECORD / "frame_cells.parquet"
    if not frame_path.exists():
        raise FileNotFoundError(
            f"WP-05 FULL run of record frame missing: {frame_path} "
            "(CITYHOURS reuses it verbatim rather than rebuilding the raster frame)"
        )
    return pd.read_parquet(frame_path)


def select_pilot_tiles(frame_df: pd.DataFrame, origin_x: float, origin_y: float, target_fraction: float) -> list[tuple[int, int]]:
    """Stratified-by-tile-size pilot: build_s is a per-TILE cost (measured
    dominant in wp05_pilot's own extrapolation note), so a pilot that samples
    a FRACTION OF CELLS scattered across nearly every tile (as the plain
    9-stratum cell draw would) still forces building nearly every tile's
    surface — it does not save the thing that is expensive. Instead this
    samples whole TILES, spread across the tile-size distribution (deciles),
    until the sampled tiles' total cell count reaches target_fraction of the
    frame — a stratified draw over the one axis (tile size) that actually
    varies per-tile cost.
    """
    i, j = tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), origin_x, origin_y)
    tile_sizes = pd.Series(1, index=pd.MultiIndex.from_arrays([i, j])).groupby(level=[0, 1]).sum()
    tile_sizes = tile_sizes.sort_values()
    n_tiles = len(tile_sizes)
    target_cells = int(round(target_fraction * len(frame_df)))

    deciles = np.linspace(0, n_tiles - 1, num=min(10, n_tiles)).astype(int)
    ordered = list(tile_sizes.index)
    picked: list[tuple[int, int]] = []
    picked_cells = 0
    d_idx = 0
    seen = set()
    while picked_cells < target_cells and len(seen) < n_tiles:
        tile_key = ordered[deciles[d_idx % len(deciles)]]
        d_idx += 1
        if tile_key in seen:
            # decile pointer exhausted at this stride; fall back to the next
            # unseen tile by size rank so the pilot still reaches its target.
            remaining = [t for t in ordered if t not in seen]
            if not remaining:
                break
            tile_key = remaining[len(remaining) // 2]
        seen.add(tile_key)
        picked.append(tile_key)
        picked_cells += int(tile_sizes.loc[tile_key])
    return picked


# ---------------------------------------------------------------------------
# Orchestration: run a given tile list, checkpointed
# ---------------------------------------------------------------------------

def run_tile_pass(
    tiles: list[tuple[int, int]],
    frame_df: pd.DataFrame,
    run_dir: Path,
    tile_subdir: str,
    *,
    dtm_path: Path,
    footprints_path: Path,
    sky: "wp02_sky.CumulativeSky",
    directions: np.ndarray,
    device: str,
    meta: dict,
    reference_days: dict,
    duration_thresholds_h: list[int],
    patch_az_deg: np.ndarray,
    origin_x: float,
    origin_y: float,
) -> dict:
    tile_dir = run_dir / tile_subdir
    tile_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    timing_path = run_dir / f"timing_{tile_subdir}.jsonl"

    i, j = tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), origin_x, origin_y)
    frame_df = frame_df.assign(_tile_i=i, _tile_j=j)

    started = time.perf_counter()
    with rasterio.open(dtm_path) as dtm_dataset:
        for tile_i, tile_j in tiles:
            tile_id = f"{tile_i}_{tile_j}"
            out_path = tile_dir / f"tile_{tile_id}.parquet"
            if out_path.exists():
                continue
            obs = frame_df[(frame_df["_tile_i"] == tile_i) & (frame_df["_tile_j"] == tile_j)]
            obs = obs.drop(columns=["_tile_i", "_tile_j"])
            out, timing = evaluate_tile_hours(
                dtm_dataset=dtm_dataset, footprints_path=footprints_path,
                tile_i=tile_i, tile_j=tile_j, origin_x=origin_x, origin_y=origin_y,
                cell_m=CELL_M, obs_df=obs, directions=directions, sky=sky, device=device,
                tmp_dir=tmp_dir, meta=meta, reference_days=reference_days,
                duration_thresholds_h=duration_thresholds_h, patch_az_deg=patch_az_deg,
            )
            out.to_parquet(out_path, index=False)
            with timing_path.open("a") as fh:
                fh.write(json.dumps({"_utc": _utc_now(), **timing.__dict__}) + "\n")
            n_done = sum(1 for _ in tile_dir.glob("tile_*.parquet"))
            print(
                f"[cityhours/{tile_subdir}] tile {tile_id} done: n_obs={timing.n_obs} "
                f"build_s={timing.build_s:.1f} engine_s={timing.engine_s:.1f} "
                f"hours_s={timing.hours_s:.2f} peak_gb={timing.peak_gb:.3f} "
                f"({n_done}/{len(tiles)} tiles) elapsed_s={time.perf_counter() - started:.0f}",
                flush=True,
            )

    return {
        "n_tiles_total": len(tiles),
        "n_tiles_done": sum(1 for _ in tile_dir.glob("tile_*.parquet")),
        "wall_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Pilot extrapolation (same per-tile + per-cell cost model wp05_pilot uses,
# extended with the hours_s term CITYHOURS adds on top of build_s/engine_s)
# ---------------------------------------------------------------------------

def compute_pilot_extrapolation(run_dir: Path, n_active_tiles_in_frame: int, city_total_cells: int) -> dict:
    timing_path = run_dir / "timing_pilot_tiles.jsonl"
    rows = [json.loads(l) for l in timing_path.read_text().splitlines() if l.strip()]
    n_obs = sum(r["n_obs"] for r in rows)
    n_tiles = len(rows)
    build_s_total = sum(r["build_s"] for r in rows)
    engine_s_total = sum(r["engine_s"] for r in rows)
    hours_s_total = sum(r["hours_s"] for r in rows)
    peaks = [r["peak_gb"] for r in rows if r["peak_gb"] == r["peak_gb"]]
    peak_gb = max(peaks) if peaks else float("nan")

    build_s_per_tile = build_s_total / n_tiles if n_tiles else float("nan")
    engine_s_per_cell = engine_s_total / n_obs if n_obs else float("nan")
    hours_s_per_cell = hours_s_total / n_obs if n_obs else float("nan")

    proj_hours = (
        n_active_tiles_in_frame * build_s_per_tile
        + city_total_cells * (engine_s_per_cell + hours_s_per_cell)
    ) / 3600.0

    return {
        "_utc": _utc_now(),
        "n_pilot_tiles": n_tiles,
        "n_pilot_cells": n_obs,
        "pilot_fraction_of_frame": n_obs / city_total_cells if city_total_cells else float("nan"),
        "measured_build_s_total": build_s_total,
        "measured_engine_s_total": engine_s_total,
        "measured_hours_s_total": hours_s_total,
        "build_s_per_tile_measured": build_s_per_tile,
        "engine_s_per_cell_measured": engine_s_per_cell,
        "hours_s_per_cell_measured": hours_s_per_cell,
        "peak_gb_measured": peak_gb,
        "n_active_tiles_in_frame": n_active_tiles_in_frame,
        "city_total_cells": city_total_cells,
        "projected_hours_full_run": proj_hours,
        "go_no_go": "GO" if proj_hours <= 3.0 else "STOP — projected wall time exceeds ~3h",
        "note": (
            "Per-tile model, same shape as wp05_pilot.compute_extrapolation: "
            "build_s is charged once per tile against the tile ceiling "
            "(n_active_tiles_in_frame, the SAME 325 measured by the WP-05 FULL "
            "run of record's own frame), engine_s and hours_s (the two costs "
            "CITYHOURS adds/changes over the WP-05 FULL run: return_horizon=True "
            "and direct_sun_hours on both reference days) are charged per cell "
            "against the full 8,402,056-cell frame."
        ),
    }


# ---------------------------------------------------------------------------
# Reproduction check: route (b)'s svf/kwh_m2 vs the run of record's
# ---------------------------------------------------------------------------

def reproduction_check(consolidated: pd.DataFrame, run_of_record_parquet: Path, corr_min: float = 0.9999, max_abs_diff_max: float = 0.01) -> dict:
    record = pd.read_parquet(run_of_record_parquet, columns=["row", "col", "svf", "kwh_m2"])
    merged = consolidated[["row", "col", "svf", "kwh_m2"]].merge(
        record, on=["row", "col"], suffixes=("_b", "_record")
    )
    result = {"_utc": _utc_now(), "n_matched": int(len(merged)), "n_consolidated": int(len(consolidated)), "n_record": int(len(record))}
    passed = True
    for metric in ("svf", "kwh_m2"):
        a = merged[f"{metric}_b"].to_numpy(dtype="float64")
        b = merged[f"{metric}_record"].to_numpy(dtype="float64")
        diff = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
        max_abs_diff = float(np.max(diff)) if len(diff) else float("nan")
        ok = corr >= corr_min and max_abs_diff <= max_abs_diff_max
        passed = passed and ok
        result[metric] = {
            "corr": corr, "max_abs_diff": max_abs_diff,
            "corr_min": corr_min, "max_abs_diff_max": max_abs_diff_max, "pass": ok,
        }
    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# Route (a): sun-hours recovered from the STORED BINARY MASK — cross-check
# only, never published (L1: route (a) is never a published number).
# ---------------------------------------------------------------------------

def _sun_vector_from_positions(alt_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    alt = np.radians(alt_deg)
    az = np.radians(az_deg)
    return np.column_stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])


def route_a_hours_for_day(
    vis_packed: np.ndarray, n_patches: int, directions: np.ndarray,
    date_str: str, meta: dict, duration_thresholds_h: list[int], chunk: int = 200_000,
) -> dict:
    """Sun-hours from the coarse binary mask alone: at each sampled time, the
    NEAREST patch by full 3-D direction (not just azimuth — the mask carries
    no continuous horizon to compare an altitude against, only per-patch
    visible/not-visible) stands in for "sun visible". Coarser than route (b)
    by construction — a ~11-deg patch quantises both azimuth and altitude,
    where route (b)'s horizon is continuous in azimuth. Cross-check only.
    """
    from .wp04_sites import sun_positions

    hourly = sun_positions(date_str, meta, "1h")
    fine = sun_positions(date_str, meta, "10min")
    hourly_alt, hourly_az = hourly["apparent_elevation"].to_numpy(), hourly["azimuth"].to_numpy()
    fine_alt, fine_az = fine["apparent_elevation"].to_numpy(), fine["azimuth"].to_numpy()
    hourly_daylight = hourly_alt > 0.0
    fine_daylight = fine_alt > 0.0

    hourly_vec = _sun_vector_from_positions(hourly_alt, hourly_az)
    fine_vec = _sun_vector_from_positions(fine_alt, fine_az)
    hourly_patch = np.argmax(directions @ hourly_vec.T, axis=0)
    fine_patch = np.argmax(directions @ fine_vec.T, axis=0)

    n = vis_packed.shape[0]
    hours_fractional = np.zeros(n, dtype=np.float32)
    thresholds = {k: np.zeros(n, dtype=bool) for k in duration_thresholds_h}

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        vis_chunk = unpack_visibility(vis_packed[start:end], n_patches)
        vis_hour = vis_chunk[:, hourly_patch] & hourly_daylight[None, :]
        vis_fine = vis_chunk[:, fine_patch] & fine_daylight[None, :]
        frac = vis_fine.sum(axis=1).astype(np.float32) / 6.0
        hours_fractional[start:end] = frac
        for k in duration_thresholds_h:
            thresholds[k][start:end] = frac >= k

    result = {"hours_fractional": hours_fractional}
    for k in duration_thresholds_h:
        result[f"ge_{k}h"] = thresholds[k]
    return result


def route_a_vs_b_lossiness(
    run_dir: Path, consolidated_b: pd.DataFrame, run_of_record_parquet: Path,
    directions: np.ndarray, meta: dict, reference_days: dict, duration_thresholds_h: list[int],
) -> dict:
    """Cross-check only (spec: "(a) is never a published number") — reads
    the run of record's stored visibility_packed mask, computes route (a)
    sun-hours from it, and reports correlation + median abs diff against
    route (b)'s continuous-horizon hours on the SAME cells. Written to
    lossiness_check.json in runs/cityhours_<UTC>/, never to summary.json.
    """
    record = pd.read_parquet(run_of_record_parquet, columns=["row", "col", "visibility_packed"])
    merged = consolidated_b[["row", "col"] + [f"hours_{l}" for l in reference_days]].merge(
        record, on=["row", "col"]
    )
    vis_packed = np.vstack([np.frombuffer(b, dtype=np.uint8) for b in merged["visibility_packed"]])

    result = {"_utc": _utc_now(), "n_matched": int(len(merged)), "never_published": True}
    for label, date_str in reference_days.items():
        route_a = route_a_hours_for_day(vis_packed, int(P1_SKY_PATCHES), directions, date_str, meta, duration_thresholds_h)
        a = route_a["hours_fractional"]
        b = merged[f"hours_{label}"].to_numpy(dtype="float32")
        diff = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
        result[label] = {
            "corr_a_vs_b": corr,
            "median_abs_diff_a_vs_b_hours": float(np.median(diff)),
            "p95_abs_diff_a_vs_b_hours": float(np.percentile(diff, 95)),
        }
    return result


# ---------------------------------------------------------------------------
# summary.json — publishable-candidate only (per-cell layer stays in runs/)
# ---------------------------------------------------------------------------

def build_summary(consolidated: pd.DataFrame, favelas_gdf: gpd.GeoDataFrame, reference_days: dict, duration_thresholds_h: list[int]) -> dict:
    result = {
        "_utc": _utc_now(),
        "status": PROVISIONAL_STATUS,
        "sky_patches": int(P1_SKY_PATCHES),
        "n_cells": int(len(consolidated)),
        "reference_days": reference_days,
        "duration_thresholds_h": duration_thresholds_h,
        "citywide": {"status": PROVISIONAL_STATUS},
        "study_favelas": {},
    }
    for label in reference_days:
        citywide_hours = consolidated[f"hours_{label}"].to_numpy()
        result["citywide"][f"sun_h_{label}"] = quantile_block(citywide_hours, qs=QUANTILES)
        shares = {}
        for k in duration_thresholds_h:
            shares[f"share_ge_{k}h"] = float(consolidated[f"ge_{k}h_{label}"].mean())
        result["citywide"][f"share_ge_{label}"] = shares

    for name in STUDY_FAVELAS:
        matched, method = match_favela_group(favelas_gdf, name)
        cod_favela_ids = matched["cod_favela"].astype(int).tolist() if len(matched) else []
        sub = consolidated[consolidated["favela_id"].isin(cod_favela_ids)] if cod_favela_ids else consolidated.iloc[0:0]
        entry = {"status": PROVISIONAL_STATUS, "match_method": method}
        for label in reference_days:
            entry[f"sun_h_{label}"] = favela_summary(sub[f"hours_{label}"].to_numpy(), consolidated[f"hours_{label}"].to_numpy())
        result["study_favelas"][name] = entry

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _common_inputs(data_root: Path, params: dict):
    dtm_path = data_root / params["terrain"]["dtm_city"]
    footprints_path = data_root / params["footprints"]["canonical_layer"].split()[0]
    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"
    epw_path = data_root / params["weather"]["primary_epw"]
    return dtm_path, footprints_path, favelas_path, epw_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    params = load_params()
    reference_days = {
        "winter_solstice": params["reference_days"]["winter_solstice"],
        "equinox": params["reference_days"]["equinox"],
    }
    duration_thresholds_h = params["reference_days"]["duration_thresholds_h"]

    dtm_path, footprints_path, favelas_path, epw_path = _common_inputs(data_root, params)

    run_dir = Path(args.run_dir) if args.run_dir else data_root / "runs" / (
        f"cityhours_{args.mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    frame_df = load_frame(data_root)
    with rasterio.open(dtm_path) as src:
        origin_x, origin_y = src.bounds.left, src.bounds.bottom

    directions, _weights = generate_tregenza_patches()
    sky = wp02_sky.build(epw_path)
    meta = epw_meta(epw_path)
    patch_az_deg = patch_azimuth_deg(directions)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_tiles = sorted(set(zip(
        *tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), origin_x, origin_y)
    )))
    n_active_tiles_in_frame = len(all_tiles)

    common = dict(
        dtm_path=dtm_path, footprints_path=footprints_path, sky=sky, directions=directions,
        device=device, meta=meta, reference_days=reference_days,
        duration_thresholds_h=duration_thresholds_h, patch_az_deg=patch_az_deg,
        origin_x=origin_x, origin_y=origin_y,
    )

    if args.mode == "pilot":
        pilot_tiles = select_pilot_tiles(frame_df, origin_x, origin_y, PILOT_TARGET_FRACTION)
        (run_dir / "pilot_tiles.json").write_text(json.dumps([f"{i}_{j}" for i, j in pilot_tiles], indent=1))
        pass_report = run_tile_pass(pilot_tiles, frame_df, run_dir, "pilot_tiles", **common)
        extrapolation = compute_pilot_extrapolation(run_dir, n_active_tiles_in_frame, len(frame_df))
        (run_dir / "extrapolation.json").write_text(json.dumps(extrapolation, indent=1))
        (run_dir / "manifest.json").write_text(json.dumps({
            "_utc": _utc_now(),
            "mode": "pilot",
            "sky": {"patches": int(P1_SKY_PATCHES)},
            "pilot_rule": "config/params.yaml#/sampling/pilot_rule",
            "pilot_target_fraction": PILOT_TARGET_FRACTION,
            "frame_source": f"runs/{WP05_FULL_RUN_OF_RECORD}/frame_cells.parquet (reused verbatim)",
            "device": device,
            "torch_version": torch.__version__,
            "git_sha": _git_sha(),
            "n_active_tiles_in_frame": n_active_tiles_in_frame,
        }, indent=1))
        print(json.dumps({"run_dir": str(run_dir), "pass": pass_report, "extrapolation": extrapolation}, indent=1))
        return 0

    # full
    t_start = time.perf_counter()
    pass_report = run_tile_pass(all_tiles, frame_df, run_dir, "tiles", **common)
    wall_s = time.perf_counter() - t_start

    consolidated = consolidate_tiles(run_dir / "tiles", run_dir / "cityhours_full.parquet")

    run_of_record_parquet = data_root / "runs" / WP05_FULL_RUN_OF_RECORD / "wp05_full.parquet"
    reproduction = reproduction_check(consolidated, run_of_record_parquet)
    (run_dir / "reproduction_check.json").write_text(json.dumps(reproduction, indent=1))

    lossiness = route_a_vs_b_lossiness(
        run_dir, consolidated, run_of_record_parquet, directions, meta, reference_days, duration_thresholds_h
    )
    (run_dir / "lossiness_check.json").write_text(json.dumps(lossiness, indent=1))

    favelas_gdf = gpd.read_file(favelas_path)
    summary = build_summary(consolidated, favelas_gdf, reference_days, duration_thresholds_h)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    manifest = {
        "_utc": _utc_now(),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_m": CELL_M,
        "max_dist_m": HALO_M,
        "tile_m": TILE_M,
        "march_sampling": "nearest",
        "frame_source": f"runs/{WP05_FULL_RUN_OF_RECORD}/frame_cells.parquet (reused verbatim)",
        "device": device,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "n_tiles_total": pass_report["n_tiles_total"],
        "n_tiles_done": pass_report["n_tiles_done"],
        "wall_s": wall_s,
        "n_cells_consolidated": int(len(consolidated)),
        "reproduction_check_pass": reproduction["pass"],
        "run_design_status": PROVISIONAL_STATUS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(json.dumps({
        "run_dir": str(run_dir), "pass": pass_report, "wall_s": wall_s,
        "n_cells": len(consolidated), "reproduction_pass": reproduction["pass"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

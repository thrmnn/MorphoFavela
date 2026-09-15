"""WP-05: citywide FULL run — exhaustive mode (every frame cell, 1 m surface).

Spec: docs/wp05_full_spec.md. Reuses wp05_pilot's sampling-frame, tile/halo,
visibility-packing and single-tile-evaluation primitives unmodified
(build_frame, compute_slope_deg, assign_strata, evaluate_tile, tile
indexing) and adds only what exhaustive mode needs on top: every frame cell
as an observer (no stratified draw), a per-cell favela_id, tile checkpoints
under runs/wp05_full_<UTC>/tiles/, consolidation, and the citywide
distribution report.

Release class (docs/wp05_full_spec.md "Release class"): the per-cell
citywide layer is compute-but-withhold (red line L1). This module never
writes a map figure; only distribution.json (quantiles + the five favelas'
positions) is the deliverable that may travel.

Run: python -m src.brisa_solar.wp05_full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize
from scipy.stats import percentileofscore

from . import wp02_sky
from .constants import P1_SKY_PATCHES, load_params
from .wp05_pilot import (
    HALO_M,
    TILE_M,
    TileTiming,
    _git_sha,
    _utc_now,
    assign_strata,
    build_frame,
    compute_slope_deg,
    evaluate_tile,
    tile_index_for_xy,
)
from src.svf_v2.compute import generate_tregenza_patches

FULL_CELL_M = 1.0
SKY_MODEL_LABEL = "epw_weighted"

#: The spec's five study favelas, matched against Favelas_Limit_2019's own
#: `nome`/`complexo` fields — see match_favela_group.
STUDY_FAVELAS = ["Vidigal", "Rocinha", "Complexo do Alemão", "Maré", "Rio das Pedras"]

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Spec: "Every number carries status: PROVISIONAL — wp05_run_design
#: untapped" — the run design (A_exhaustive_1m vs a PI-tapped 3M sample) is
#: a /ops card, not yet a PI-signed decision.
PROVISIONAL_STATUS = "PROVISIONAL — wp05_run_design untapped"


# ---------------------------------------------------------------------------
# Per-cell favela_id (rasterized, not sjoin — 8.4M points against 1074
# polygons is a spatial-join a CPU shouldn't be asked to do; a burn onto the
# same grid the frame is already indexed against is exact and O(cells)).
# ---------------------------------------------------------------------------

def rasterize_favela_id(favelas: gpd.GeoDataFrame, transform, shape: tuple[int, int]) -> np.ndarray:
    """Burn cod_favela (1..1165 in Favelas_Limit_2019, never 0) onto the DTM
    grid; 0 = the cell is not inside any favela polygon."""
    shapes = list(zip(favelas.geometry, favelas["cod_favela"].astype("int32")))
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def _norm(s) -> str:
    return str(s).strip().lower()


def match_favela_group(favelas: gpd.GeoDataFrame, target: str) -> tuple[gpd.GeoDataFrame, str]:
    """Match one study-favela name to Favelas_Limit_2019 polygons.

    Two-tier rule, checked in order:
      1. exact match on `complexo` — the layer's own agglomeration field.
         Complexo do Alemão, Maré and Rio das Pedras are each split across
         several polygons under one complexo name; this is the authoritative
         grouping for them.
      2. exact match on `nome`, only used when (1) found nothing — this
         covers Vidigal and Rocinha, which are "Isolada" (standalone, no
         complexo). Required to be unique or it is flagged, never guessed.

    Deliberately never a substring/contains match: "Rocinha" as a substring
    also matches "Matinha (RA - Rocinha)", a different polygon (verified
    2026-09-15 against the live Favelas_Limit_2019.shp) — contains-matching
    would silently pull in a neighbourhood reference, not the favela itself.
    Returns (matched_polygons, method) where method is one of
    "complexo_exact", "nome_exact_unique", "nome_exact_AMBIGUOUS", "no_match".
    """
    nt = _norm(target)
    by_complexo = favelas[favelas["complexo"].map(_norm) == nt]
    if len(by_complexo) > 0:
        return by_complexo, "complexo_exact"
    by_nome = favelas[favelas["nome"].map(_norm) == nt]
    if len(by_nome) == 1:
        return by_nome, "nome_exact_unique"
    if len(by_nome) > 1:
        return by_nome, "nome_exact_AMBIGUOUS"
    return favelas.iloc[0:0], "no_match"


def _polygon_records(matched: gpd.GeoDataFrame) -> list[dict]:
    return [
        {
            "objectid": int(r.objectid),
            "cod_favela": int(r.cod_favela),
            "nome": str(r.nome),
            "complexo": str(r.complexo),
        }
        for r in matched.itertuples()
    ]


# ---------------------------------------------------------------------------
# Sampling frame -> every frame cell as an observer (no stratified draw)
# ---------------------------------------------------------------------------

def observers_from_frame(
    in_frame: np.ndarray,
    stratum: np.ndarray,
    favela_id_raster: np.ndarray,
    transform,
) -> pd.DataFrame:
    """Every in-frame cell becomes one observer row. Pure array op — the
    exhaustive-mode analogue of wp05_pilot.stratified_pilot's draw, except
    there is no draw: `len(df) == frame["frame_cells"]` by construction.
    """
    rows, cols = np.where(in_frame)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    return pd.DataFrame({
        "row": rows,
        "col": cols,
        "x": xs,
        "y": ys,
        "stratum": stratum[rows, cols],
        "favela_id": favela_id_raster[rows, cols].astype("int32"),
    })


def build_exhaustive_frame(
    run_dir: Path, *, dtm_path: Path, footprints_path: Path, favelas_path: Path,
) -> pd.DataFrame:
    """Build the full sampling frame (same rules as the pilot) and turn every
    frame cell into an observer, with a per-cell favela_id. Writes
    frame_cells.parquet + frame_diagnostics.json (including the five
    study-favela matches, so the match is auditable independent of the run).
    """
    from .wp02_surface import build_surface, load_surface  # deferred: heavy import, only needed here

    params = load_params()
    domain = params["domain"]

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

    favelas = gpd.read_file(favelas_path)
    if favelas.crs is not None and str(favelas.crs) != str(crs):
        favelas = favelas.to_crs(crs)
    favela_id_raster = rasterize_favela_id(favelas, transform, dtm.shape)

    frame_df = observers_from_frame(frame["in_frame"], stratum, favela_id_raster, transform)

    # Same geography-bounded tile ceiling the pilot's extrapolation used
    # (kept here so a resumed/second run can cross-check it against the
    # pilot's n_active_2km_tiles_in_frame=325 without recomputing the frame).
    fi, fj = tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), bounds.left, bounds.bottom)
    n_active_tiles_in_frame = int(len(set(zip(fi.tolist(), fj.tolist()))))

    run_dir.mkdir(parents=True, exist_ok=True)
    frame_df.to_parquet(run_dir / "frame_cells.parquet", index=False)

    favela_diag = {}
    for name in STUDY_FAVELAS:
        matched, method = match_favela_group(favelas, name)
        favela_diag[name] = {
            "match_method": method,
            "n_polygons_matched": int(len(matched)),
            "matched_polygons": _polygon_records(matched),
        }

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
            "n_active_2km_tiles_in_frame": n_active_tiles_in_frame,
        },
        "favela_rasterize": {
            "n_polygons": int(len(favelas)),
            "cod_favela_min": int(favelas["cod_favela"].min()),
            "cod_favela_max": int(favelas["cod_favela"].max()),
            "n_frame_cells_with_favela_id": int((frame_df["favela_id"] > 0).sum()),
        },
        "study_favela_matches": favela_diag,
        "git_sha": _git_sha(),
    }
    (run_dir / "frame_diagnostics.json").write_text(json.dumps(diagnostics, indent=1))
    return frame_df


# ---------------------------------------------------------------------------
# Tile pass: every frame cell, 1 m surface, nearest-cell march (evaluate_tile
# already hardcodes march_sampling="nearest" — unmodified from the pilot)
# ---------------------------------------------------------------------------

def run_exhaustive_pass(
    frame_df: pd.DataFrame,
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
    """Evaluate every frame cell at 1 m, tiled, checkpointed per tile under
    run_dir/tiles/. Resumable: a tile whose checkpoint parquet already
    exists is skipped.
    """
    tile_dir = run_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp"
    timing_path = run_dir / "timing_full.jsonl"

    i, j = tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), origin_x, origin_y)
    frame_df = frame_df.assign(_tile_i=i, _tile_j=j)
    tiles = sorted(frame_df.groupby(["_tile_i", "_tile_j"]).groups.keys())

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
            obs = frame_df[(frame_df["_tile_i"] == tile_i) & (frame_df["_tile_j"] == tile_j)]
            obs = obs.drop(columns=["_tile_i", "_tile_j"])
            out, timing = evaluate_tile(
                dtm_dataset=dtm_dataset, footprints_path=footprints_path,
                tile_i=tile_i, tile_j=tile_j, origin_x=origin_x, origin_y=origin_y,
                cell_m=FULL_CELL_M, obs_df=obs, directions=directions, sky=sky,
                device=device, tmp_dir=tmp_dir,
            )
            out = out.rename(columns={"tile_id": "tile", "irradiation_kwh_m2": "kwh_m2"})
            out["sky_model"] = SKY_MODEL_LABEL
            out.to_parquet(out_path, index=False)
            with timing_path.open("a") as fh:
                fh.write(json.dumps({"_utc": _utc_now(), **timing.__dict__}) + "\n")
            n_done = sum(1 for _ in tile_dir.glob("tile_*.parquet"))
            print(
                f"[full] tile {tile_id} done: n_obs={timing.n_obs} "
                f"build_s={timing.build_s:.1f} engine_s={timing.engine_s:.1f} "
                f"peak_gb={timing.peak_gb:.3f} ({n_done}/{len(tiles)} tiles) "
                f"elapsed_s={time.perf_counter() - started:.0f}",
                flush=True,
            )

    return {
        "n_tiles_total": len(tiles),
        "n_tiles_done": sum(1 for _ in tile_dir.glob("tile_*.parquet")),
        "stopped_early": stopped_early,
        "wall_s": time.perf_counter() - started,
    }


def consolidate_tiles(tile_dir: Path, out_path: Path) -> pd.DataFrame:
    """Reassemble per-tile checkpoints into one Parquet. Raises if any (x, y)
    pair appears more than once — tiling partitions the frame exclusively
    (tile_index_for_xy is a deterministic floor-division), so a duplicate
    means two tiles double-counted a cell, not an expected outcome.
    """
    tile_files = sorted(Path(tile_dir).glob("tile_*.parquet"))
    if not tile_files:
        raise FileNotFoundError(f"no tile checkpoints found in {tile_dir}")
    merged = pd.concat([pd.read_parquet(f) for f in tile_files], ignore_index=True)
    n_dup = int(merged.duplicated(subset=["x", "y"]).sum())
    if n_dup:
        raise ValueError(f"{n_dup} duplicate (x, y) rows across {len(tile_files)} tile checkpoints")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return merged


# ---------------------------------------------------------------------------
# distribution.json
# ---------------------------------------------------------------------------

def quantile_block(values: np.ndarray, qs=QUANTILES) -> dict:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    block = {"status": PROVISIONAL_STATUS, "n": int(len(values))}
    if len(values) == 0:
        block.update({f"p{round(q * 100)}": None for q in qs})
        return block
    qvals = np.quantile(values, qs)
    block.update({f"p{round(q * 100)}": float(v) for q, v in zip(qs, qvals)})
    return block


def favela_summary(values: np.ndarray, citywide_values: np.ndarray) -> dict:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "status": PROVISIONAL_STATUS, "n": 0,
            "median": None, "iqr": None, "citywide_percentile_position": None,
        }
    median = float(np.median(values))
    q25, q75 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    pct = float(percentileofscore(citywide_values, median, kind="mean"))
    return {
        "status": PROVISIONAL_STATUS,
        "n": int(len(values)),
        "median": median,
        "iqr": [q25, q75],
        "citywide_percentile_position": pct,
    }


def compute_distribution(consolidated: pd.DataFrame, favelas: gpd.GeoDataFrame) -> dict:
    citywide_svf = consolidated["svf"].to_numpy()
    citywide_kwh = consolidated["kwh_m2"].to_numpy()

    result = {
        "_utc": _utc_now(),
        "status": PROVISIONAL_STATUS,
        "sky_model": SKY_MODEL_LABEL,
        "sky_patches": int(P1_SKY_PATCHES),
        "n_cells": int(len(consolidated)),
        "citywide": {
            "status": PROVISIONAL_STATUS,
            "svf": quantile_block(citywide_svf),
            "kwh_m2": quantile_block(citywide_kwh),
        },
        "per_stratum": {},
        "study_favelas": {},
    }

    for s, grp in consolidated.groupby("stratum"):
        result["per_stratum"][str(int(s))] = {
            "status": PROVISIONAL_STATUS,
            "n": int(len(grp)),
            "svf": quantile_block(grp["svf"].to_numpy()),
            "kwh_m2": quantile_block(grp["kwh_m2"].to_numpy()),
        }

    for name in STUDY_FAVELAS:
        matched, method = match_favela_group(favelas, name)
        cod_favela_ids = matched["cod_favela"].astype(int).tolist() if len(matched) else []
        sub = consolidated[consolidated["favela_id"].isin(cod_favela_ids)] if cod_favela_ids else consolidated.iloc[0:0]
        result["study_favelas"][name] = {
            "status": PROVISIONAL_STATUS,
            "match_method": method,
            "matched_polygons": _polygon_records(matched),
            "svf": favela_summary(sub["svf"].to_numpy(), citywide_svf),
            "kwh_m2": favela_summary(sub["kwh_m2"].to_numpy(), citywide_kwh),
        }

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="defaults to a fresh runs/wp05_full_<UTC>/ in the main checkout")
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela", help="main checkout, where data/ lives")
    ap.add_argument("--time-budget-hours", type=float, default=None, help="stop the tile pass after this many hours (resumable)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    params = load_params()

    run_dir = Path(args.run_dir) if args.run_dir else data_root / "runs" / ("wp05_full_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = data_root / params["terrain"]["dtm_city"]
    footprints_path = data_root / params["footprints"]["canonical_layer"].split()[0]
    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"
    epw_path = data_root / params["weather"]["primary_epw"]

    frame_parquet = run_dir / "frame_cells.parquet"
    if frame_parquet.exists():
        frame_df = pd.read_parquet(frame_parquet)
    else:
        frame_df = build_exhaustive_frame(run_dir, dtm_path=dtm_path, footprints_path=footprints_path, favelas_path=favelas_path)

    with rasterio.open(dtm_path) as src:
        origin_x, origin_y = src.bounds.left, src.bounds.bottom

    directions, _weights = generate_tregenza_patches()
    sky = wp02_sky.build(epw_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_budget_s = args.time_budget_hours * 3600.0 if args.time_budget_hours else None

    t_start = time.perf_counter()
    pass_report = run_exhaustive_pass(
        frame_df, run_dir,
        dtm_path=dtm_path, footprints_path=footprints_path,
        sky=sky, directions=directions, device=device,
        origin_x=origin_x, origin_y=origin_y, time_budget_s=time_budget_s,
    )
    wall_s = time.perf_counter() - t_start

    consolidated = consolidate_tiles(run_dir / "tiles", run_dir / "wp05_full.parquet")

    peak_gb = float("nan")
    timing_path = run_dir / "timing_full.jsonl"
    if timing_path.exists():
        rows = [json.loads(l) for l in timing_path.read_text().splitlines() if l.strip()]
        peaks = [r["peak_gb"] for r in rows if r["peak_gb"] == r["peak_gb"]]
        peak_gb = max(peaks) if peaks else float("nan")

    sky_section = json.dumps(params["sky"], sort_keys=True)
    manifest = {
        "_utc": _utc_now(),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "cell_m": FULL_CELL_M,
        "obs_height_m": 1.5,
        "max_dist_m": HALO_M,
        "tile_m": TILE_M,
        "march_sampling": "nearest",
        "sampling_rule": f"WP-05 FULL exhaustive: {len(frame_df)} frame cells (every fabric cell), no sampling error",
        "device": device,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        "params_sky_section_sha256": hashlib.sha256(sky_section.encode()).hexdigest()[:16],
        "n_tiles_total": pass_report["n_tiles_total"],
        "n_tiles_done": pass_report["n_tiles_done"],
        "stopped_early": pass_report["stopped_early"],
        "wall_s": wall_s,
        "peak_gb": peak_gb,
        "n_cells_consolidated": int(len(consolidated)),
        "run_design_status": PROVISIONAL_STATUS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    favelas_gdf = gpd.read_file(favelas_path)
    distribution = compute_distribution(consolidated, favelas_gdf)
    (run_dir / "distribution.json").write_text(json.dumps(distribution, indent=1))

    print(json.dumps({
        "run_dir": str(run_dir), "pass": pass_report, "wall_s": wall_s,
        "peak_gb": peak_gb, "n_cells": len(consolidated),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

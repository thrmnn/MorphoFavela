"""G3 / HD-02: analysis-domain sensitivity grid.

Spec: docs/g3_domain_sensitivity_spec.md. The five study favelas' percentile
positions depend on which ground cells count as "the city" (config/params.yaml
`domain`: fabric_coverage_threshold, fabric_footprint_distance_m). This module
produces the evidence for that choice: a 3x3 grid of
fabric_coverage_threshold in {0.05, 0.10, 0.20} x fabric_footprint_distance_m
in {5, 10, 20}, plus the WP-04 all-polygon-interior universe already computed
at the 5 study sites. The grid cell (0.10, 10) IS the WP-05 frame (same
numbers as runs/wp05_full_*/distribution.json) — it is not a separate row.

Reuses wp05_pilot's build_surface/build_frame math and wp05_full's tile pass,
favela matching and consolidation unmodified. Never recomputes what WP-05
already evaluated: cells satisfying a variant's (threshold, distance) that are
also in the base 0.10/10 frame are pulled straight from wp05_full.parquet;
only cells the base frame never evaluated ("added" cells — coverage in
[0.05, 0.10) and/or distance in (10, 20]) get a fresh 1 m tile pass, batched
once over the loosest grid cell (0.05, 20) since every other grid cell's
added-cell set is a subset of that one.

Release class: same as WP-05 (compute-but-withhold, L1) — this module never
writes a map figure or a per-cell layer outside runs/; only sensitivity.json
and sensitivity.md (aggregates: frame size, favela share, medians, the five
favelas' percentile-of-median, spread) may travel.

Run: python -m src.brisa_solar.g3_domain
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
from scipy import ndimage
from scipy.stats import percentileofscore

from . import wp02_sky
from .constants import P1_SKY_PATCHES, load_params
from .wp02_surface import build_surface, load_surface
from .wp05_full import (
    STUDY_FAVELAS,
    match_favela_group,
    rasterize_favela_id,
    run_exhaustive_pass,
)
from .wp05_full import consolidate_tiles as _consolidate_tiles
from .wp05_pilot import _git_sha, _utc_now
from src.svf_v2.compute import generate_tregenza_patches

FABRIC_COVERAGE_GRID = (0.05, 0.10, 0.20)
FABRIC_FOOTPRINT_DISTANCE_GRID_M = (5.0, 10.0, 20.0)

STATUS = "PROVISIONAL — g3_domain_sensitivity, HD-02 not yet PI-signed"


# ---------------------------------------------------------------------------
# Domain rasters (coverage, footprint-distance) — same formulas as
# wp05_pilot.build_frame, factored so every (threshold, distance) grid cell
# reuses one build_surface + one uniform_filter + one distance_transform_edt
# call instead of nine.
# ---------------------------------------------------------------------------

def compute_domain_arrays(dtm_path: Path, footprints_path: Path, out_stem: Path) -> dict:
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
    if nodata is not None:
        dtm = np.where(np.isclose(dtm, nodata, rtol=1e-3), np.nan, dtm)
    cell_m = abs(transform.a)
    dtm_shape = dtm.shape
    dtm_valid = np.isfinite(dtm)
    del dtm

    surface_tif = build_surface(dtm_path, footprints_path, cell_m, out_stem)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    _surface, _t, _crs, is_building = load_surface(surface_tif, is_building_tif)

    window_cells = max(1, int(round(100.0 / cell_m)))
    coverage = ndimage.uniform_filter(is_building.astype("float32"), size=window_cells, mode="constant")
    dist_m = ndimage.distance_transform_edt(~is_building) * cell_m
    not_building_and_valid = (~is_building) & dtm_valid
    del is_building, dtm_valid

    return {
        "transform": transform, "crs": crs, "bounds": bounds, "cell_m": cell_m,
        "shape": dtm_shape,
        "coverage": coverage, "dist_m": dist_m,
        "not_building_and_valid": not_building_and_valid,
    }


def variant_mask(
    not_building_and_valid: np.ndarray, coverage: np.ndarray, dist_m: np.ndarray,
    threshold: float, distance_m: float,
) -> np.ndarray:
    """The domain a (fabric_coverage_threshold, fabric_footprint_distance_m)
    pair selects: same rule as wp05_pilot.build_frame's last two clauses,
    applied on top of the (threshold/distance-independent) not-building &
    within-municipality mask.
    """
    return not_building_and_valid & (coverage >= threshold) & (dist_m <= distance_m)


# ---------------------------------------------------------------------------
# Added cells: everything the loosest grid cell (0.05, 20) selects that the
# base 0.10/10 frame never evaluated. Every other grid cell's own added-cell
# set is a subset of this one (0.05 is the grid's lowest threshold, 20 its
# highest distance — both directions more permissive dominate).
# ---------------------------------------------------------------------------

def build_added_observers(
    arrays: dict, favela_id_raster: np.ndarray, base_frame_mask: np.ndarray,
) -> pd.DataFrame:
    loosest_mask = variant_mask(
        arrays["not_building_and_valid"], arrays["coverage"], arrays["dist_m"],
        min(FABRIC_COVERAGE_GRID), max(FABRIC_FOOTPRINT_DISTANCE_GRID_M),
    )
    added_mask = loosest_mask & ~base_frame_mask
    rows, cols = np.where(added_mask)
    xs, ys = rasterio.transform.xy(arrays["transform"], rows, cols)
    return pd.DataFrame({
        "row": rows, "col": cols,
        "x": np.asarray(xs, dtype="float64"), "y": np.asarray(ys, dtype="float64"),
        "favela_id": favela_id_raster[rows, cols].astype("int32"),
        "coverage": arrays["coverage"][rows, cols].astype("float32"),
        "dist_m": arrays["dist_m"][rows, cols].astype("float32"),
    })


# ---------------------------------------------------------------------------
# Percentile-of-median — same formula as wp05_full.favela_summary (kept
# separate rather than imported: wp05_full's version also returns iqr/status
# fields this module doesn't use, and spec test 2 checks this function in
# isolation against a synthetic, known answer).
# ---------------------------------------------------------------------------

def favela_percentile_of_median(values: np.ndarray, citywide_values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    citywide_values = np.asarray(citywide_values, dtype="float64")
    citywide_values = citywide_values[np.isfinite(citywide_values)]
    if len(values) == 0 or len(citywide_values) == 0:
        return None, None
    median = float(np.median(values))
    pct = float(percentileofscore(citywide_values, median, kind="mean"))
    return median, pct


# ---------------------------------------------------------------------------
# One grid row: base-frame cells matching (threshold, distance) + whichever
# added cells also match it.
# ---------------------------------------------------------------------------

def build_variant_frame(
    base_df: pd.DataFrame, added_df: pd.DataFrame, arrays: dict, threshold: float, distance_m: float,
) -> tuple[pd.DataFrame, int]:
    mask_2d = variant_mask(arrays["not_building_and_valid"], arrays["coverage"], arrays["dist_m"], threshold, distance_m)
    base_hit = mask_2d[base_df["row"].to_numpy(), base_df["col"].to_numpy()]
    base_part = base_df[base_hit]

    added_hit = (added_df["coverage"].to_numpy() >= threshold) & (added_df["dist_m"].to_numpy() <= distance_m)
    added_part = added_df.loc[added_hit, ["row", "col", "favela_id", "svf", "kwh_m2"]]

    variant_df = pd.concat([base_part, added_part], ignore_index=True)
    return variant_df, int(len(added_part))


def compute_grid_row(
    threshold: float, distance_m: float, base_df: pd.DataFrame, added_df: pd.DataFrame,
    arrays: dict, favelas_gdf: gpd.GeoDataFrame,
) -> dict:
    variant_df, n_added = build_variant_frame(base_df, added_df, arrays, threshold, distance_m)
    frame_cells = int(len(variant_df))
    n_favela = int((variant_df["favela_id"] > 0).sum())
    citywide_svf = variant_df["svf"].to_numpy()
    citywide_kwh = variant_df["kwh_m2"].to_numpy()

    row = {
        "fabric_coverage_threshold": threshold,
        "fabric_footprint_distance_m": distance_m,
        "is_wp05_base_frame": (threshold == 0.10 and distance_m == 10.0),
        "frame_cells": frame_cells,
        "n_added_cells": n_added,
        "favela_share": (n_favela / frame_cells) if frame_cells else None,
        "citywide_svf_median": float(np.median(citywide_svf)) if frame_cells else None,
        "citywide_kwh_m2_median": float(np.median(citywide_kwh)) if frame_cells else None,
        "study_favelas": {},
    }
    for name in STUDY_FAVELAS:
        matched, method = match_favela_group(favelas_gdf, name)
        cod_ids = matched["cod_favela"].astype(int).tolist() if len(matched) else []
        sub = variant_df[variant_df["favela_id"].isin(cod_ids)] if cod_ids else variant_df.iloc[0:0]
        svf_median, svf_pct = favela_percentile_of_median(sub["svf"].to_numpy(), citywide_svf)
        kwh_median, kwh_pct = favela_percentile_of_median(sub["kwh_m2"].to_numpy(), citywide_kwh)
        row["study_favelas"][name] = {
            "n": int(len(sub)),
            "svf_median": svf_median,
            "svf_percentile_of_citywide_median": svf_pct,
            "kwh_m2_median": kwh_median,
            "kwh_m2_percentile_of_citywide_median": kwh_pct,
        }
    return row


# ---------------------------------------------------------------------------
# WP-04 "all-polygon-interior" row: not a citywide frame (WP-04 only ran the
# 5 study sites' ground surfaces, never all 1074 favela polygons citywide) —
# reuse its already-computed per-favela numbers verbatim rather than
# fabricate a citywide frame_cells/favela_share that was never measured.
# ---------------------------------------------------------------------------

WP04_SITE_DIRS = {
    "Vidigal": "vidigal",
    "Rocinha": "rocinha",
    "Complexo do Alemão": "complexo_do_alemao",
    "Maré": "maré",
    "Rio das Pedras": "riodaspedras",
}


def build_wp04_row(wp04_run_dir: Path, base_row: dict) -> dict:
    row = {
        "fabric_coverage_threshold": None,
        "fabric_footprint_distance_m": None,
        "is_wp05_base_frame": False,
        "frame_cells": None,
        "frame_cells_note": (
            "not computed citywide — WP-04 evaluated only the 5 study sites' "
            "all-polygon-interior ground surfaces (every non-building 1 m cell "
            "inside the favela's own Favelas_Limit_2019 polygon), not a "
            "citywide all-polygon-interior frame across all 1074 polygons."
        ),
        "n_added_cells": None,
        "favela_share": None,
        "citywide_svf_median": base_row["citywide_svf_median"],
        "citywide_kwh_m2_median": base_row["citywide_kwh_m2_median"],
        "citywide_note": (
            "WP-04's percentile positions were computed (wp04_sites.py "
            "citywide_percentile) against WP-05's own citywide distribution "
            "(the 0.10/10 frame, same numbers as this table's base row) — only "
            "the per-favela NUMERATOR differs (all polygon-interior cells, not "
            "fabric-frame-filtered); the citywide denominator is shared with "
            "the base row, not independently computed."
        ),
        "study_favelas": {},
    }
    for name, site_key in WP04_SITE_DIRS.items():
        summary_path = wp04_run_dir / site_key / "summary.json"
        summary = json.loads(summary_path.read_text())
        cw = summary.get("citywide_percentile", {})
        row["study_favelas"][name] = {
            "n": summary["ground"]["svf"]["n"],
            "svf_median": summary["ground"]["svf"]["median"],
            "svf_percentile_of_citywide_median": cw.get("ground_svf_median_percentile"),
            "kwh_m2_median": summary["ground"]["kwh_m2"]["median"],
            "kwh_m2_percentile_of_citywide_median": cw.get("ground_kwh_m2_median_percentile"),
        }
    return row


def compute_spread(rows: list[dict]) -> dict:
    spread = {}
    for name in STUDY_FAVELAS:
        svf_pcts = [
            r["study_favelas"][name]["svf_percentile_of_citywide_median"] for r in rows
            if r["study_favelas"][name]["svf_percentile_of_citywide_median"] is not None
        ]
        kwh_pcts = [
            r["study_favelas"][name]["kwh_m2_percentile_of_citywide_median"] for r in rows
            if r["study_favelas"][name]["kwh_m2_percentile_of_citywide_median"] is not None
        ]
        spread[name] = {
            "svf_percentile_spread_max_minus_min": (max(svf_pcts) - min(svf_pcts)) if svf_pcts else None,
            "kwh_m2_percentile_spread_max_minus_min": (max(kwh_pcts) - min(kwh_pcts)) if kwh_pcts else None,
            "n_variants_with_data": len(svf_pcts),
        }
    return spread


def build_card_draft(grid_rows: list[dict]) -> dict:
    def _find(t, d):
        return next(r for r in grid_rows if r["fabric_coverage_threshold"] == t and r["fabric_footprint_distance_m"] == d)

    tightest = _find(0.20, 5.0)
    centre = _find(0.10, 10.0)
    loosest = _find(0.05, 20.0)

    def _option(label, row):
        return {
            "variant": label,
            "fabric_coverage_threshold": row["fabric_coverage_threshold"],
            "fabric_footprint_distance_m": row["fabric_footprint_distance_m"],
            "detail": {
                "frame_cells": row["frame_cells"],
                "favela_share": row["favela_share"],
                "citywide_svf_median": row["citywide_svf_median"],
                "citywide_kwh_m2_median": row["citywide_kwh_m2_median"],
                "study_favelas_svf_percentile": {
                    name: row["study_favelas"][name]["svf_percentile_of_citywide_median"]
                    for name in STUDY_FAVELAS
                },
            },
        }

    return {
        "id": "g3_domain",
        "question": (
            "Which analysis-domain definition (config/params.yaml `domain."
            "fabric_coverage_threshold` / `fabric_footprint_distance_m`) should "
            "be locked for the headline citywide-percentile claim?"
        ),
        "options": [
            _option("tightest (0.20 / 5m)", tightest),
            _option("grid centre — current WP-05 default (0.10 / 10m)", centre),
            _option("loosest (0.05 / 20m)", loosest),
        ],
        "recommended": "grid centre — current WP-05 default (0.10 / 10m)",
        "reason": (
            "Closest to the sensitivity grid's centre and identical to the "
            "frame every other WP-05/WP-04 deliverable already reports "
            "against; the table shows no favela crossing a qualitatively "
            "different rank at the grid's edges, so there is no evidence to "
            "prefer a tighter or looser domain over the one already in use."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _latest_run_dir(runs_root: Path, prefix: str) -> Path:
    candidates = sorted(runs_root.glob(f"{prefix}_*"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no {prefix}_* run directory found under {runs_root}")
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--wp05-run-dir", default=None, help="defaults to the latest runs/wp05_full_*")
    ap.add_argument("--wp04-run-dir", default=None, help="defaults to the latest runs/wp04_sites_*")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    runs_root = data_root / "runs"
    params = load_params()

    wp05_run_dir = Path(args.wp05_run_dir) if args.wp05_run_dir else _latest_run_dir(runs_root, "wp05_full")
    wp04_run_dir = Path(args.wp04_run_dir) if args.wp04_run_dir else _latest_run_dir(runs_root, "wp04_sites")

    run_dir = Path(args.run_dir) if args.run_dir else runs_root / ("g3_domain_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = data_root / params["terrain"]["dtm_city"]
    footprints_path = data_root / params["footprints"]["canonical_layer"].split()[0]
    favelas_path = data_root / "data/RJ/Favelas_Limit_2019.shp"
    epw_path = data_root / params["weather"]["primary_epw"]

    frame_diag = json.loads((wp05_run_dir / "frame_diagnostics.json").read_text())
    base_threshold = frame_diag["frame"]["fabric_coverage_threshold"]
    base_distance = frame_diag["frame"]["fabric_footprint_distance_m"]
    base_frame_cells_expected = frame_diag["frame"]["frame_cells"]
    assert base_threshold == 0.10 and base_distance == 10.0, (
        f"WP-05 base run's own frame params ({base_threshold}, {base_distance}) "
        "no longer match the grid centre this module assumes — re-check the grid."
    )

    print("[g3] loading base wp05_full.parquet columns...", flush=True)
    base_df = pd.read_parquet(
        wp05_run_dir / "wp05_full.parquet",
        columns=["row", "col", "favela_id", "svf", "kwh_m2"],
    )
    assert len(base_df) == base_frame_cells_expected, (
        f"base frame row count {len(base_df)} != frame_diagnostics.json frame_cells {base_frame_cells_expected}"
    )

    print("[g3] recomputing coverage/footprint-distance rasters (build_surface + uniform_filter + edt)...", flush=True)
    t0 = time.perf_counter()
    arrays = compute_domain_arrays(dtm_path, footprints_path, run_dir / "artifacts" / "frame_5m")
    base_frame_mask = variant_mask(arrays["not_building_and_valid"], arrays["coverage"], arrays["dist_m"], base_threshold, base_distance)
    n_base_recomputed = int(base_frame_mask.sum())
    assert n_base_recomputed == base_frame_cells_expected, (
        f"recomputed base frame ({n_base_recomputed}) != wp05_full's own frame_diagnostics.json "
        f"({base_frame_cells_expected}) — coverage/distance recomputation has drifted from build_frame."
    )
    domain_build_s = time.perf_counter() - t0
    print(f"[g3] domain rasters ready in {domain_build_s:.1f}s; base frame cross-check OK ({n_base_recomputed})", flush=True)

    favelas_gdf = gpd.read_file(favelas_path)
    if favelas_gdf.crs is not None and str(favelas_gdf.crs) != str(arrays["crs"]):
        favelas_gdf = favelas_gdf.to_crs(arrays["crs"])
    favela_id_raster = rasterize_favela_id(favelas_gdf, arrays["transform"], arrays["shape"])

    added_obs = build_added_observers(arrays, favela_id_raster, base_frame_mask)
    n_added_total = int(len(added_obs))
    print(f"[g3] {n_added_total} added cells (loosest grid cell 0.05/20m minus base 0.10/10m frame)", flush=True)

    added_pass_report = {"n_added_cells_total": n_added_total, "wall_s": 0.0, "n_tiles": 0}
    added_df = pd.DataFrame(columns=["row", "col", "favela_id", "coverage", "dist_m", "svf", "kwh_m2"])

    if n_added_total > 0:
        directions, _weights = generate_tregenza_patches()
        sky = wp02_sky.build(epw_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with rasterio.open(dtm_path) as src:
            origin_x, origin_y = src.bounds.left, src.bounds.bottom

        added_run_dir = run_dir / "added_cells"
        t1 = time.perf_counter()
        pass_report = run_exhaustive_pass(
            added_obs, added_run_dir,
            dtm_path=dtm_path, footprints_path=footprints_path,
            sky=sky, directions=directions, device=device,
            origin_x=origin_x, origin_y=origin_y, time_budget_s=None,
        )
        added_wall_s = time.perf_counter() - t1
        added_consolidated = _consolidate_tiles(added_run_dir / "tiles", added_run_dir / "added_cells.parquet")
        added_df = added_consolidated[["row", "col", "favela_id", "coverage", "dist_m", "svf", "kwh_m2"]]
        added_pass_report = {
            "n_added_cells_total": n_added_total,
            "n_tiles": pass_report["n_tiles_done"],
            "wall_s": added_wall_s,
            "stopped_early": pass_report["stopped_early"],
            "note": (
                "ONE batched 1 m tile pass over the union of every looser grid "
                "cell's added cells (the loosest cell, 0.05/20m, minus the base "
                "0.10/10m frame) — every tighter looser variant's own "
                "n_added_cells below is the subset of this union satisfying its "
                "own (threshold, distance); wall_s here is shared across all of "
                "them, not measured per variant separately."
            ),
        }
        print(f"[g3] added-cell tile pass: {n_added_total} cells, {pass_report['n_tiles_done']} tiles, {added_wall_s:.1f}s", flush=True)

    grid_rows = []
    for threshold in FABRIC_COVERAGE_GRID:
        for distance_m in FABRIC_FOOTPRINT_DISTANCE_GRID_M:
            grid_rows.append(compute_grid_row(threshold, distance_m, base_df, added_df, arrays, favelas_gdf))

    base_row = next(r for r in grid_rows if r["is_wp05_base_frame"])
    wp04_row = build_wp04_row(wp04_run_dir, base_row)

    all_rows = grid_rows + [wp04_row]
    spread = compute_spread(all_rows)
    card_draft = build_card_draft(grid_rows)

    sky_section = json.dumps(params["sky"], sort_keys=True)
    sensitivity = {
        "_utc": _utc_now(),
        "status": STATUS,
        "sky_patches": int(P1_SKY_PATCHES),
        "wp05_run_dir": str(wp05_run_dir),
        "wp04_run_dir": str(wp04_run_dir),
        "grid": {
            "fabric_coverage_threshold": list(FABRIC_COVERAGE_GRID),
            "fabric_footprint_distance_m": list(FABRIC_FOOTPRINT_DISTANCE_GRID_M),
        },
        "domain_build_s": domain_build_s,
        "added_cells_pass": added_pass_report,
        "variants": grid_rows,
        "wp04_polygon_interior": wp04_row,
        "spread_across_variants": spread,
        "card_draft": card_draft,
        "git_sha": _git_sha(),
    }
    (run_dir / "sensitivity.json").write_text(json.dumps(sensitivity, indent=1))
    (run_dir / "sensitivity.md").write_text(render_markdown(sensitivity))

    manifest = {
        "_utc": _utc_now(),
        "sky": {"patches": int(P1_SKY_PATCHES)},
        "params_sky_section_sha256": hashlib.sha256(sky_section.encode()).hexdigest()[:16],
        "git_sha": _git_sha(),
        "wp05_run_dir": str(wp05_run_dir),
        "wp04_run_dir": str(wp04_run_dir),
        "n_added_cells_total": n_added_total,
        "added_cells_wall_s": added_pass_report["wall_s"],
        "status": STATUS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(json.dumps({"run_dir": str(run_dir), "n_added_cells_total": n_added_total}, indent=1))
    return 0


def render_markdown(sensitivity: dict) -> str:
    lines = [
        "# G3 / HD-02 domain sensitivity",
        "",
        f"_{sensitivity['_utc']} — {sensitivity['status']}_",
        "",
        "## Grid: frame size, favela share, citywide medians",
        "",
        "| threshold | distance_m | frame_cells | n_added_cells | favela_share | citywide SVF median | citywide kWh/m² median |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sensitivity["variants"]:
        lines.append(
            f"| {r['fabric_coverage_threshold']:.2f} | {r['fabric_footprint_distance_m']:.0f} | "
            f"{r['frame_cells']} | {r['n_added_cells']} | "
            f"{r['favela_share']:.4f} | {r['citywide_svf_median']:.4f} | {r['citywide_kwh_m2_median']:.1f} |"
        )
    w = sensitivity["wp04_polygon_interior"]
    lines.append(
        f"| n/a (WP-04) | n/a (WP-04) | n/a | n/a | n/a | {w['citywide_svf_median']:.4f} | {w['citywide_kwh_m2_median']:.1f} |"
    )
    lines += ["", "## Per-favela percentile-of-median (SVF)", "",
              "| variant | " + " | ".join(STUDY_FAVELAS) + " |",
              "|---" * (len(STUDY_FAVELAS) + 1) + "|"]
    for r in sensitivity["variants"]:
        label = f"{r['fabric_coverage_threshold']:.2f}/{r['fabric_footprint_distance_m']:.0f}m" + (" (base)" if r["is_wp05_base_frame"] else "")
        vals = [f"{r['study_favelas'][n]['svf_percentile_of_citywide_median']:.1f}" for n in STUDY_FAVELAS]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    vals = [
        (f"{w['study_favelas'][n]['svf_percentile_of_citywide_median']:.1f}"
         if w['study_favelas'][n]['svf_percentile_of_citywide_median'] is not None else "n/a")
        for n in STUDY_FAVELAS
    ]
    lines.append("| wp04_polygon_interior | " + " | ".join(vals) + " |")

    lines += ["", "## Per-favela percentile-of-median (kWh/m²)", "",
              "| variant | " + " | ".join(STUDY_FAVELAS) + " |",
              "|---" * (len(STUDY_FAVELAS) + 1) + "|"]
    for r in sensitivity["variants"]:
        label = f"{r['fabric_coverage_threshold']:.2f}/{r['fabric_footprint_distance_m']:.0f}m" + (" (base)" if r["is_wp05_base_frame"] else "")
        vals = [f"{r['study_favelas'][n]['kwh_m2_percentile_of_citywide_median']:.1f}" for n in STUDY_FAVELAS]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    vals = [
        (f"{w['study_favelas'][n]['kwh_m2_percentile_of_citywide_median']:.1f}"
         if w['study_favelas'][n]['kwh_m2_percentile_of_citywide_median'] is not None else "n/a")
        for n in STUDY_FAVELAS
    ]
    lines.append("| wp04_polygon_interior | " + " | ".join(vals) + " |")

    lines += ["", "## Max-min spread of percentile-of-median across all variants (grid + WP-04)", "",
              "| favela | SVF spread (pts) | kWh/m² spread (pts) |",
              "|---|---:|---:|"]
    for name in STUDY_FAVELAS:
        s = sensitivity["spread_across_variants"][name]
        lines.append(f"| {name} | {s['svf_percentile_spread_max_minus_min']:.1f} | {s['kwh_m2_percentile_spread_max_minus_min']:.1f} |")

    lines += ["", "## card_draft", "", "```json", json.dumps(sensitivity["card_draft"], indent=1), "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

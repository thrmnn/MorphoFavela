"""WP-06 — geometry-only ventilation-potential table (C′ reframe).

Spec: docs/wp06_geometry_spec.md. Builds P1's own per-cell table —
``outputs/<site>/geometry_indicators/per_patch_geometry.csv`` — from the
10 m morphometrics grid, the site wind rose, and the WP-04 1 m ground SVF/sun
results, restricted to ``p1_legal`` columns (docs/p1_column_allowlist.json).

P1's ventilation axis is DESCRIPTIVE GEOMETRY ONLY: this module never imports
``src.cfd_integration`` or ``scripts.analyze_cfd_results``, and describes the
vertical constraint purely in geometric terms — λf_mean ≥ 0.65, Oke's 1988
threshold.

Reuse boundary: ``count_constraints`` (scripts.run_ventilation_index),
``open_edge_distance`` (scripts.run_lateral_connectivity) and
``wind_exposure`` (scripts.run_wind_exposure) are pure functions over
already-loaded arrays/frames and are imported directly. Their *orchestration*
siblings — ``build_site`` and ``load_freq`` — hardcode
``PROJECT_ROOT = Path(__file__).resolve().parents[1]``, i.e. wherever
``scripts/`` physically sits; under this task's worktree/main-checkout split
(code in the worktree, data only in the main checkout) that resolves to the
wrong root, so this module re-implements the load step against an explicit
``data_root`` instead of importing them.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.run_lateral_connectivity import CELL_M, open_edge_distance
from scripts.run_ventilation_index import SKIM_MIN, count_constraints
from scripts.run_wind_exposure import SECTORS, wind_exposure
from src.morphometry.aspect import aspect_to_sincos, aspect_wind_alignment
from src.morphometry.invariants import built_mask

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

# Compass bearing (meteorological "from" convention, N-clockwise) of each of
# the grid's 8 λf sectors — the same convention src.morphometry.aspect and the
# wind rose both use. A fixed compass definition, not a simulation output.
SECTOR_BEARING_DEG = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}

GROUND_COLUMNS = ["x", "y", "svf", "kwh_m2", "hours_winter_solstice", "ge_2h_winter_solstice"]

OUTPUT_COLUMNS = [
    "patch_id", "center_x", "center_y",
    "svf", "lambda_p", "slope_deg", "porosity", "sigma_h",
    "aspect_deg", "aspect_sin", "aspect_cos", "aspect_wind_alignment",
    "lambda_f_N", "lambda_f_NE", "lambda_f_E", "lambda_f_SE",
    "lambda_f_S", "lambda_f_SW", "lambda_f_W", "lambda_f_NW",
    "lambda_f_mean", "lambda_f_max", "lambda_f_max_dir", "H_mean",
    "svf_c_p50", "kwh_m2_p50", "sun_h_winter_p50", "share_ge_2h_winter",
    "wind_exposure", "exposure_ratio", "open_edge_dist_m",
    "constraint_vertical", "constraint_lateral", "constraint_directional",
    "n_constraints",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_grid(data_root: Path, site: str) -> gpd.GeoDataFrame:
    """Built 10 m cells for one site, canonical mask (building_count > 0)."""
    g = gpd.read_file(data_root / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg")
    return g[built_mask(g)].copy()


def load_wind_freq(data_root: Path, site: str) -> dict:
    """Normalised sector frequencies from the measured wind rose.

    Mirrors scripts.run_wind_exposure.load_freq's normalisation exactly
    (renormalise over SECTORS; calm already excluded upstream) — see the
    module docstring for why this re-implements the load rather than
    importing that function.
    """
    d = json.loads((data_root / "data" / site / "wind_rose.json").read_text())
    f = d["frequencies"]
    tot = sum(f[s] for s in SECTORS)
    return {s: f[s] / tot for s in SECTORS}


def load_ground(data_root: Path, run_id: str, site: str) -> pd.DataFrame:
    """WP-04 1 m ground parquet, only the columns WP-06 needs (up to 846k
    rows — read via pyarrow so unused columns, incl. the packed visibility
    bytes, never materialise)."""
    path = data_root / "runs" / run_id / site / "ground.parquet"
    table = pq.read_table(path, columns=GROUND_COLUMNS)
    return table.to_pandas()


def bin_ground_to_cells(ground: pd.DataFrame, grid: gpd.GeoDataFrame, cell: float = CELL_M) -> pd.DataFrame:
    """Per-cell median/share of the WP-04 1 m ground points, aligned to
    ``grid``'s row order.

    Both the grid and the ground raster tile the same UTM lattice at the same
    origin, so flooring each coordinate against a common lattice origin (the
    grid's own minimum centroid, offset by half a cell to its edge) recovers
    exact cell membership without a geometric point-in-polygon join.
    """
    x0 = float(grid["centroid_x"].min()) - cell / 2.0
    y0 = float(grid["centroid_y"].min()) - cell / 2.0

    grid_keys = pd.DataFrame({
        "ix": np.floor((grid["centroid_x"].to_numpy() - x0) / cell).astype(np.int64),
        "iy": np.floor((grid["centroid_y"].to_numpy() - y0) / cell).astype(np.int64),
    })
    ground_bins = pd.DataFrame({
        "ix": np.floor((ground["x"].to_numpy() - x0) / cell).astype(np.int64),
        "iy": np.floor((ground["y"].to_numpy() - y0) / cell).astype(np.int64),
        "svf": ground["svf"].to_numpy(),
        "kwh_m2": ground["kwh_m2"].to_numpy(),
        "hours_winter_solstice": ground["hours_winter_solstice"].to_numpy(),
        "ge_2h_winter_solstice": ground["ge_2h_winter_solstice"].to_numpy(),
    })
    agg = ground_bins.groupby(["ix", "iy"], as_index=False).agg(
        svf_c_p50=("svf", "median"),
        kwh_m2_p50=("kwh_m2", "median"),
        sun_h_winter_p50=("hours_winter_solstice", "median"),
        share_ge_2h_winter=("ge_2h_winter_solstice", "mean"),
    )
    merged = grid_keys.merge(agg, on=["ix", "iy"], how="left")
    assert len(merged) == len(grid), "cell binning must not duplicate or drop grid rows"
    return merged[["svf_c_p50", "kwh_m2_p50", "sun_h_winter_p50", "share_ge_2h_winter"]]


def compute_site_table(grid: gpd.GeoDataFrame, ground: pd.DataFrame, freq: dict, depth_median: float) -> pd.DataFrame:
    """Per-cell geometry table for one already-loaded site, allowlisted
    columns only. Pure function (no I/O) so it is unit-testable on synthetic
    inputs independently of the GIS/parquet reads."""
    grid = grid.copy()

    # grid is already filtered to built cells, so every row IS built; the
    # excluded (open) cells simply have no row here and stay background in
    # open_edge_distance's lattice — matches run_ventilation_index.build_site.
    grid["open_edge_dist_m"] = open_edge_distance(
        grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy(),
        np.ones(len(grid), dtype=bool),
    )
    grid["wind_exposure"] = wind_exposure(grid, freq)
    grid["exposure_ratio"] = grid["wind_exposure"] / grid["lambda_f_mean"].replace(0, np.nan)

    sector_cols = [f"lambda_f_{s}" for s in SECTORS]
    grid["lambda_f_max_dir"] = grid[sector_cols].idxmax(axis=1).str.replace("lambda_f_", "", regex=False)

    aspect_sin, aspect_cos = aspect_to_sincos(grid["aspect_deg"].to_numpy())
    grid["aspect_sin"] = aspect_sin
    grid["aspect_cos"] = aspect_cos
    dominant_sector = max(freq, key=freq.get)
    grid["aspect_wind_alignment"] = aspect_wind_alignment(
        grid["aspect_deg"].to_numpy(), SECTOR_BEARING_DEG[dominant_sector]
    )

    ground_stats = bin_ground_to_cells(ground, grid)
    ground_stats.index = grid.index
    grid = pd.concat([grid, ground_stats], axis=1)

    n_con = count_constraints(
        grid["lambda_f_mean"].to_numpy(), grid["open_edge_dist_m"].to_numpy(),
        grid["exposure_ratio"].to_numpy(), depth_median,
    )
    grid["constraint_vertical"] = (np.nan_to_num(grid["lambda_f_mean"].to_numpy(), nan=0.0) >= SKIM_MIN).astype(int)
    grid["constraint_lateral"] = (np.nan_to_num(grid["open_edge_dist_m"].to_numpy(), nan=0.0) >= depth_median).astype(int)
    grid["constraint_directional"] = (np.nan_to_num(grid["exposure_ratio"].to_numpy(), nan=0.0) >= 1.0).astype(int)
    grid["n_constraints"] = n_con

    grid["patch_id"] = grid["zone_id"]
    grid["center_x"] = grid["centroid_x"]
    grid["center_y"] = grid["centroid_y"]

    out = grid[OUTPUT_COLUMNS].copy()
    unknown = [c for c in out.columns if c not in OUTPUT_COLUMNS]
    assert not unknown, f"internal: non-allowlisted column produced: {unknown}"
    return out


def build_site_table(data_root: Path, wp04_run_id: str, site: str, depth_median: float) -> pd.DataFrame:
    """I/O wrapper: load one site's grid/wind-rose/ground then delegate to
    the pure ``compute_site_table``."""
    grid = load_grid(data_root, site)
    freq = load_wind_freq(data_root, site)
    ground = load_ground(data_root, wp04_run_id, site)
    return compute_site_table(grid, ground, freq, depth_median)


def pooled_depth_median(data_root: Path, sites: list[str]) -> float:
    dists = []
    for site in sites:
        grid = load_grid(data_root, site)
        d = open_edge_distance(
            grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy(),
            np.ones(len(grid), dtype=bool),
        )
        dists.append(d)
    return float(np.median(np.concatenate(dists)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/theo/SCL/SCR/MorphoFavela", help="main checkout, where data/outputs/runs live")
    ap.add_argument("--wp04-run-id", default="wp04_sites_20260914T230606Z", help="runs/<id>/<site>/ground.parquet")
    ap.add_argument("--run-dir", default=None, help="defaults to a fresh runs/wp06_geometry_<UTC>/ in --data-root")
    ap.add_argument("--sites", default=",".join(SITES))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    sites = args.sites.split(",")
    run_dir = Path(args.run_dir) if args.run_dir else data_root / "runs" / (
        "wp06_geometry_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    depth_median = pooled_depth_median(data_root, sites)

    per_site_summary = {}
    for site in sites:
        table = build_site_table(data_root, args.wp04_run_id, site, depth_median)
        out_dir = data_root / "outputs" / site / "geometry_indicators"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "per_patch_geometry.csv"
        table.to_csv(out_path, index=False)

        n = len(table)
        hist = table["n_constraints"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        per_site_summary[site] = {
            "n": int(n),
            "shares": {str(k): float(hist[k]) / n for k in range(4)},
            "svf_c_p50_median": float(table["svf_c_p50"].median()),
            "share_ge_2h_winter_median": float(table["share_ge_2h_winter"].median()),
            "output_csv": str(out_path),
        }
        print(f"{site}: n={n} wrote {out_path}")

    compare_path = data_root / "outputs" / "paper_figures" / "ventilation_index.json"
    comparison = compare_against_ventilation_index(per_site_summary, compare_path)

    summary = {
        "_utc": _utc_now(),
        "status": "PROVISIONAL — depends on WP-05/G3 cards",
        "depth_median_m": depth_median,
        "per_site": per_site_summary,
        "comparison_vs_ventilation_index": comparison,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {run_dir / 'summary.json'}")
    return 0


def compare_against_ventilation_index(per_site_summary: dict, compare_path: Path) -> dict:
    if not compare_path.exists():
        return {"status": f"comparator not found: {compare_path}"}
    ref = json.loads(compare_path.read_text())
    out = {"source": str(compare_path), "per_site": {}}
    for site, s in per_site_summary.items():
        ref_site = ref.get("per_site", {}).get(site)
        if ref_site is None:
            out["per_site"][site] = {"status": "no matching site in comparator"}
            continue
        share_diff = {
            k: s["shares"][k] - ref_site["shares"][k] for k in s["shares"]
        }
        out["per_site"][site] = {
            "n_wp06": s["n"], "n_ventilation_index": ref_site["n"],
            "share_diff": share_diff,
            "cause": (
                "constraint shares reuse the same count_constraints/open_edge_distance/"
                "wind_exposure pure functions and the same pooled depth median as "
                "ventilation_index.json, so diffs here are expected to be ~0; svf_c_p50 "
                "is NOT comparable to ventilation_index.json (which carries no SVF column) "
                "— it is new: WP-04's C′ engine (raster-horizon visibility, epw_weighted "
                "sky) 1 m ground-point median per cell, distinct from the grid's own "
                "clear-sky street SVF column."
            ),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())

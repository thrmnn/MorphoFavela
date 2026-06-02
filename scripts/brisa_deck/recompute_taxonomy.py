"""Recompute the four-state diagnostic taxonomy under THREE ventilation axes:

    (1) v2 clipped λf  → threshold pinned at Macdonald 0.40 (still saturating)
    (2) v2 clipped λf  → threshold pinned at the within-site p75
    (3) Street SVF      → SHELTERED if cell-aggregated SVF < 0.30 (Yang 2022)
    (4) H/W canyon      → SKIMMING if cell-aggregated H/W > 0.65 (Oke 1988)

Solar axis stays as in build_diagnostic_map.py (winter solar hours < 2 h).

Writes outputs/brisa_ventilation_fix/taxonomy_v2_stats.json with per-site
shares under each ventilation axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SITES = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}

SUN_THR = 2.0
LF_MACDONALD = 0.40
SVF_SHELTER = 0.30
HW_SKIM = 0.65


def aggregate_to_cells(
    grid: gpd.GeoDataFrame,
    pts_path: Path,
    col: str,
    max_radius: float = 25.0,
    k: int = 3,
) -> np.ndarray:
    pts = gpd.read_file(pts_path)[[col, "geometry"]].dropna()
    if pts.empty:
        return np.full(len(grid), np.nan)
    j = gpd.sjoin(pts, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    primary = j.groupby("zone_id")[col].median()
    out = grid["zone_id"].map(primary).values.astype(float)
    missing = np.isnan(out)
    if missing.any():
        arr = np.array([(g.x, g.y) for g in pts.geometry])
        vals = pts[col].to_numpy()
        tree = cKDTree(arr)
        cents = np.c_[grid.loc[missing, "centroid_x"].values,
                      grid.loc[missing, "centroid_y"].values]
        dists, idxs = tree.query(cents, k=min(k, len(arr)), distance_upper_bound=max_radius)
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        valid = np.isfinite(dists)
        idxs = np.where(valid, idxs, 0)
        v = vals[idxs]
        v = np.where(valid, v, np.nan)
        with np.errstate(invalid="ignore"):
            cell_vals = np.nanmedian(v, axis=1)
        out[missing] = cell_vals
    return out


def taxonomy_shares(built: np.ndarray, sun_fail: np.ndarray, vent_fail: np.ndarray) -> dict:
    valid = built & np.isfinite(sun_fail) & np.isfinite(vent_fail)
    if valid.sum() == 0:
        return {"n": 0}
    sf = sun_fail[valid].astype(bool)
    vf = vent_fail[valid].astype(bool)
    n = int(valid.sum())
    return {
        "n_classified": n,
        "adequate": float((~sf & ~vf).mean()),
        "sun_only": float((sf & ~vf).mean()),
        "vent_only": float((~sf & vf).mean()),
        "compound": float((sf & vf).mean()),
    }


def process_site(site: str, label: str) -> dict:
    grid_v2_path = _ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics_v2.gpkg"
    sol_path = _ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    svf_path = sol_path  # same file has svf + solar_hours_winter
    hw_path = _ROOT / "outputs" / site / "morphometrics" / "canyon" / "hw_streets.gpkg"

    grid = gpd.read_file(grid_v2_path)
    if "zone_id" not in grid.columns:
        grid["zone_id"] = np.arange(len(grid))
    if "centroid_x" not in grid.columns:
        c = grid.geometry.centroid
        grid["centroid_x"] = c.x
        grid["centroid_y"] = c.y

    built = ((grid.get("lambda_p", pd.Series(np.zeros(len(grid)))).fillna(0) > 0.01)
             | (grid.get("building_count", pd.Series(np.zeros(len(grid)))) > 0)).values

    sun = aggregate_to_cells(grid, sol_path, "solar_hours_winter")
    sun_fail = sun < SUN_THR

    # Ventilation axis A: v2 clipped λf with Macdonald 0.40
    lf_v2 = grid["lambda_f_mean_v2"].values.astype(float)
    vent_fail_lf_mac = lf_v2 > LF_MACDONALD

    # Ventilation axis B: v2 clipped λf with within-site p75 (relative discriminator)
    lf_built = lf_v2[built & np.isfinite(lf_v2)]
    p75 = float(np.quantile(lf_built, 0.75)) if lf_built.size else np.nan
    vent_fail_lf_p75 = lf_v2 > p75

    # Ventilation axis C: street SVF aggregated to cells, sheltered if SVF<0.30
    svf_cell = aggregate_to_cells(grid, svf_path, "svf")
    vent_fail_svf = svf_cell < SVF_SHELTER

    # Ventilation axis D: street H/W aggregated, skimming if H/W>0.65
    hw_cell = aggregate_to_cells(grid, hw_path, "HW")
    vent_fail_hw = hw_cell > HW_SKIM

    shares = {
        "lambda_f_v2_macdonald_0p40": taxonomy_shares(built, sun_fail, vent_fail_lf_mac),
        f"lambda_f_v2_p75_within_site (={p75:.3f})": taxonomy_shares(built, sun_fail, vent_fail_lf_p75),
        "svf_streets_sheltered_lt_0p30": taxonomy_shares(built, sun_fail, vent_fail_svf),
        "hw_streets_skim_gt_0p65": taxonomy_shares(built, sun_fail, vent_fail_hw),
        "legacy_lambda_f_v1_gt_0p35": None,  # left None — v1 column may be missing in some sites
    }

    # Legacy v1 if column present.
    if "lambda_f_mean_v1" in grid.columns:
        lf_v1 = grid["lambda_f_mean_v1"].values.astype(float)
        vent_fail_legacy = lf_v1 > 0.35
        shares["legacy_lambda_f_v1_gt_0p35"] = taxonomy_shares(built, sun_fail, vent_fail_legacy)

    return {"site": site, "label": label, "n_built_cells": int(built.sum()), "shares": shares}


def main() -> None:
    out_root = _ROOT / "outputs" / "brisa_ventilation_fix"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for site, label in SITES.items():
        print(f"[+] {label}")
        try:
            r = process_site(site, label)
            results.append(r)
            for axis, sh in r["shares"].items():
                if sh is None:
                    continue
                print(
                    f"    {axis:50s}  ade={100*sh['adequate']:.1f}  "
                    f"sun={100*sh['sun_only']:.1f}  vent={100*sh['vent_only']:.1f}  "
                    f"comp={100*sh['compound']:.1f}"
                )
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"site": site, "label": label, "error": str(e)})

    payload = {
        "thresholds": {
            "SUN": "winter direct sun < 2 h (WHO)",
            "LF_MACDONALD": "λf > 0.40 (Macdonald et al. 1998 WIF→SF lower bound)",
            "SVF_SHELTER": "street SVF < 0.30 (Yang et al. 2022)",
            "HW_SKIM": "H/W > 0.65 (Oke 1988 skimming onset)",
        },
        "note": (
            "Cell-aggregation of street SVF and H/W uses median-within-cell, "
            "with nearest-3 fallback within 25 m for cells without a street "
            "observation. Identical to the legacy build_diagnostic_map.py "
            "solar aggregation."
        ),
        "sites": results,
    }
    (out_root / "taxonomy_v2_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {out_root / 'taxonomy_v2_stats.json'}")


if __name__ == "__main__":
    main()

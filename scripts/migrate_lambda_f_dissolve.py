#!/usr/bin/env python3
"""Migrate every site's grid to the DISSOLVED (party-wall-corrected) λf.

User decision 2026-06-24: dissolved λf is the canonical frontal-area density
(the summed form over-counts ~1.7× in fused favela fabric). This recomputes the
8 directional λf columns + ``lambda_f_mean`` + ``lambda_f_max`` with
``dissolve=True`` and overwrites them in ``grid_metrics.gpkg`` in place,
preserving the previous summed mean/max as ``lambda_f_mean_summed`` /
``lambda_f_max_summed`` for provenance (the full summed field is regenerable by
re-running with ``dissolve=False``).

Idempotent: if ``lambda_f_mean_summed`` already exists the site is skipped unless
``--force``.

    python scripts/migrate_lambda_f_dissolve.py            # all 5 campaign sites
    python scripts/migrate_lambda_f_dissolve.py --site rocinha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.morphometry.indicators import WIND_DIRECTIONS, compute_lambda_f_directional  # noqa: E402

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
HEIGHT_CANDIDATES = ("height", "altura")


def _prep_buildings(site: str, crs) -> gpd.GeoDataFrame:
    bld = gpd.read_file(PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg")
    if bld.crs != crs:
        bld = bld.to_crs(crs)
    if "height" not in bld.columns:
        for c in HEIGHT_CANDIDATES:
            if c in bld.columns:
                bld = bld.rename(columns={c: "height"})
                break
    return bld


def migrate_site(site: str, force: bool = False) -> None:
    gpath = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(gpath)
    if "lambda_f_mean_summed" in grid.columns and not force:
        print(f"  {site}: already migrated (use --force to redo) — skipping")
        return

    bld = _prep_buildings(site, grid.crs)
    summed_mean = grid["lambda_f_mean"].copy()
    summed_max = grid["lambda_f_max"].copy()

    diss = compute_lambda_f_directional(bld, grid, clip_to_zone=True, dissolve=True)
    for label in WIND_DIRECTIONS:
        grid[f"lambda_f_{label}"] = diss[f"lambda_f_{label}"].to_numpy()
    grid["lambda_f_mean"] = diss["lambda_f_mean"].to_numpy()
    grid["lambda_f_max"] = diss["lambda_f_max"].to_numpy()
    grid["lambda_f_mean_summed"] = summed_mean.to_numpy()
    grid["lambda_f_max_summed"] = summed_max.to_numpy()

    built = grid["building_count"] > 0
    s = summed_mean[built]
    d = grid.loc[built, "lambda_f_mean"]
    print(f"  {site}: {int(built.sum()):,} built cells | "
          f"λf_mean median {s.median():.2f}→{d.median():.2f}  "
          f"p75 {s.quantile(.75):.2f}→{d.quantile(.75):.2f}  "
          f"ratio {(s / d.replace(0, float('nan'))).median():.2f}")
    grid.to_file(gpath, driver="GPKG")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None, help="single site (default: all 5)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sites = [args.site] if args.site else SITES
    for site in sites:
        migrate_site(site, force=args.force)
    print("done.")


if __name__ == "__main__":
    main()
